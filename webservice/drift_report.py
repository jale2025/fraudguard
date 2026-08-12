"""
Threshold check and background Evidently drift report for the prediction API.

The endpoints in ``app.py`` used to run the whole drift report inline. That put two
full table reads and a CPU-bound Evidently run into the request path -- on the event
loop itself for the two ``async def`` file endpoints -- and let a crashing report turn
an otherwise successful prediction into an HTTP 500.

The work is therefore split in two:

- :func:`create_report_and_trigger_workflow` stays in the request path but only reads
  the redis row counter, and schedules the rest as a FastAPI background task.
- :func:`run_drift_report` runs after the response was sent. It never raises; its
  health is reported through the ``fraudguard_drift_report_*`` metrics instead.
"""

import logging
import os
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from api_evidently_metrics import (
    DRIFT_REPORT_DURATION,
    DRIFT_REPORT_IN_PROGRESS,
    DRIFT_REPORT_RUNS,
    LAST_DRIFT_REPORT_TIMESTAMP,
)
from evidently_helper import (
    create_and_forward_data_drift_report_to_prometheus,
    get_data_from_postgresql_db,
)
from fastapi import BackgroundTasks
from redis_helper import redis_client

logger = logging.getLogger(__name__)

# Rows that have to arrive on a queue before the next report is scheduled.
THRESHOLD_QUEUE_COUNTER = 250

# Current data per queue. Keeping these here stops the prediction endpoints and the
# manual trigger endpoint from drifting apart.
QUEUE_QUERIES = {
    "not_labeled_queue": ("SELECT * FROM raw.predictions ORDER BY elapsed_sec, pc_1"),
    "labeled_queue": (
        "SELECT * FROM raw.labeled_predictions_queue ORDER BY elapsed_sec, pc_1"
    ),
}

# Baseline every queue is compared against.
REFERENCE_DATA_QUERY = (
    "SELECT * FROM dbt_prod_data_science.fct_training_data ORDER BY elapsed_sec, pc_1"
)

DEPLOYMENT_NAME = "fraud_detection_pipeline/fraud_detection_pipeline_hourly_serve"

# Rendered reports live outside /webservice on purpose: that path is bind-mounted and
# uvicorn runs with --reload, so writing there would restart the worker mid-report.
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "/reports"))

# A rendered report embeds the plots for all 30 features and runs into several MB, so
# the archive is pruned rather than kept forever.
MAX_REPORTS_PER_QUEUE = 5

# Columns that only exist on the served queues, not in the training baseline.
_SERVING_ONLY_COLUMNS = ["ingestion_time", "prediction"]

_REPORT_FILE_PREFIX = "drift_"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


# A single uvicorn worker serves the API, so an in-process flag is enough to keep two
# reports from running at once. Going multi-worker would need a redis SET NX EX lease.
_GATE = threading.Lock()
_report_in_flight = False


def claim_report_slot() -> bool:
    """
    Reserve the single drift-report slot.

    Returns:
        bool: True if the caller may schedule a report, False if one is running.

    """
    global _report_in_flight

    with _GATE:
        if _report_in_flight:
            return False

        _report_in_flight = True
        return True


def release_report_slot() -> None:
    """Free the drift-report slot. Safe to call when no slot is held."""
    global _report_in_flight

    with _GATE:
        _report_in_flight = False


def create_report_and_trigger_workflow(
    *,
    counter_name: str,
    current_data_query: str,
    background_tasks: BackgroundTasks,
) -> None:
    """
    Schedule a drift report if enough rows arrived since the last one.

    Only the redis counter read happens in the request path. Nothing here may raise:
    a monitoring hiccup must not fail an otherwise good prediction.

    Args:
        counter_name (str): Redis counter for this queue.
        current_data_query (str): SQL query for the current dataset.
        background_tasks (BackgroundTasks): Collector for the work that runs once
            the response has been sent.

    """
    try:
        redis_clt = redis_client()

        # decode_responses=True, so a present key is a str. A missing key means no
        # rows were ever counted -- nothing seeds these keys, only INCR creates them.
        raw_counter_value = redis_clt.get(name=counter_name)
        queue_counter_since_last_report = (
            int(raw_counter_value) if raw_counter_value not in (None, "") else 0
        )

        if queue_counter_since_last_report < THRESHOLD_QUEUE_COUNTER:
            return

        if not claim_report_slot():
            DRIFT_REPORT_RUNS.labels(queue=counter_name, result="skipped_busy").inc()
            logger.info(
                "A drift report is already running; skipping %s at %d rows.",
                counter_name,
                queue_counter_since_last_report,
            )
            return

        background_tasks.add_task(
            run_drift_report,
            counter_name=counter_name,
            current_data_query=current_data_query,
        )

        # Reset now rather than after the report: otherwise every request arriving
        # during the run still sees the counter above the threshold and is recorded
        # as skipped. No rows are lost, because current_data_query re-reads the whole
        # table -- the counter only decides *when* to look, not what to look at.
        redis_clt.set(name=counter_name, value=0)

    except Exception:
        logger.exception("Could not evaluate the drift trigger for %s.", counter_name)
        DRIFT_REPORT_RUNS.labels(queue=counter_name, result="counter_unavailable").inc()
        # Idempotent: a no-op unless the failure happened after the slot was claimed.
        release_report_slot()


def run_drift_report(
    *,
    counter_name: str,
    current_data_query: str,
    trigger_retraining: bool = True,
) -> None:
    """
    Run the Evidently report off the request path. Never raises.

    Args:
        counter_name (str): Queue this report describes, used as the metric label
            and in the report file name.
        current_data_query (str): SQL query for the current dataset.
        trigger_retraining (bool): Whether detected drift may start the retraining
            deployment. False for on-demand debug runs.

    """
    DRIFT_REPORT_IN_PROGRESS.set(1)

    try:
        with DRIFT_REPORT_DURATION.labels(queue=counter_name).time():
            # Read the queue as the current dataset.
            current_data = get_data_from_postgresql_db(
                db_uri=os.environ["DB_URI"], query=current_data_query
            )
            current_data = current_data.drop(
                columns=_SERVING_ONLY_COLUMNS, errors="ignore"
            )

            # Read the fct training data as the reference dataset (prod).
            reference_data = get_data_from_postgresql_db(
                db_uri=os.environ["DB_URI"], query=REFERENCE_DATA_QUERY
            )

            html_path = _report_path(counter_name)
            drift_detected = create_and_forward_data_drift_report_to_prometheus(
                reference_data=reference_data,
                current_data=current_data,
                html_path=html_path,
            )

        # Best-effort: the report's job is detecting drift and starting a retraining
        # run. Losing the HTML copy must not cost us the trigger below.
        _publish_report_file(counter_name, html_path)

        if drift_detected and trigger_retraining:
            _trigger_retraining_deployment()

        DRIFT_REPORT_RUNS.labels(queue=counter_name, result="success").inc()
        LAST_DRIFT_REPORT_TIMESTAMP.labels(queue=counter_name).set(time.time())

    except Exception:
        logger.exception("Drift report for %s failed.", counter_name)
        DRIFT_REPORT_RUNS.labels(queue=counter_name, result="failure").inc()

    finally:
        DRIFT_REPORT_IN_PROGRESS.set(0)
        release_report_slot()


def _trigger_retraining_deployment() -> None:
    """Start the retraining pipeline without waiting for it to finish."""
    # Imported lazily: it keeps this module cheap to import and stops a prefect
    # import problem from being fatal at startup.
    from prefect.deployments import run_deployment

    run_deployment(
        name=DEPLOYMENT_NAME,
        parameters={"is_triggered_by_evidently": True},
        timeout=0,
        _sync=True,
    )


def _report_path(counter_name: str) -> Path:
    """Return the destination for a new report; the name sorts chronologically."""
    stamp = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
    return REPORTS_DIR / f"{_REPORT_FILE_PREFIX}{counter_name}_{stamp}.html"


def _latest_path(counter_name: str) -> Path:
    """Return the stable path clients and Grafana link to."""
    return REPORTS_DIR / f"latest_{counter_name}.html"


def _publish_report_file(counter_name: str, html_path: Path) -> None:
    """
    Expose a finished report as ``latest_<queue>.html`` and prune the archive.

    Failures are logged and swallowed: serving the HTML is a convenience, while the
    caller still has to trigger retraining on the drift this report found.
    """
    try:
        # A copy rather than a symlink, so FileResponse works on any volume driver.
        shutil.copy2(html_path, _latest_path(counter_name))
        _prune_reports(counter_name)

    except OSError:
        logger.exception("Could not publish the drift report for %s.", counter_name)


def _prune_reports(counter_name: str) -> None:
    """Keep only the newest ``MAX_REPORTS_PER_QUEUE`` timestamped reports."""
    # The timestamp format sorts lexicographically, so plain sorting is enough.
    existing = sorted(_queue_reports(counter_name))

    for stale in existing[:-MAX_REPORTS_PER_QUEUE]:
        stale.unlink(missing_ok=True)


def _queue_reports(counter_name: str) -> list[Path]:
    """Return the timestamped report files belonging to one queue."""
    if not REPORTS_DIR.is_dir():
        return []

    return list(REPORTS_DIR.glob(f"{_REPORT_FILE_PREFIX}{counter_name}_*.html"))


def list_reports(counter_name: str | None = None) -> list[dict]:
    """
    List the retained drift reports, newest first.

    Args:
        counter_name (str | None): Restrict the listing to one queue.

    Returns:
        list[dict]: One entry per retained report holding its file name, queue,
        UTC timestamp and size in bytes.

    """
    queues = [counter_name] if counter_name is not None else list(QUEUE_QUERIES)

    reports = []
    for queue in queues:
        for path in _queue_reports(queue):
            stamp = path.stem.removeprefix(f"{_REPORT_FILE_PREFIX}{queue}_")
            reports.append(
                {
                    "filename": path.name,
                    "queue": queue,
                    "created_at": stamp,
                    "size_bytes": path.stat().st_size,
                }
            )

    return sorted(reports, key=lambda entry: entry["created_at"], reverse=True)


def report_file(filename: str) -> Path | None:
    """
    Resolve a report file name inside the reports directory.

    Args:
        filename (str): Name as returned by :func:`list_reports`.

    Returns:
        Path | None: The readable file, or None if it does not exist or would
        escape the reports directory.

    """
    candidate = (REPORTS_DIR / filename).resolve()
    reports_dir = REPORTS_DIR.resolve()

    if candidate.parent != reports_dir or not candidate.is_file():
        return None

    return candidate


def latest_report_file(counter_name: str) -> Path | None:
    """Return the newest published report for a queue, or None if there is none."""
    return report_file(_latest_path(counter_name).name)
