import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from api_evidently_metrics import (
    DATA_DRIFT_DETECTED,
    DRIFTED_COLUMNS_RATIO,
    FEATURE_DRIFT_SCORE,
    NUMBER_OF_DRIFTED_COLUMNS,
)
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

# Read environment variables
DB_URI = os.environ.get("DB_URI", "")

DRIFT_SHARE_THRESHOLD = 0.5

_DRIFTED_COUNT_TYPE = "evidently:metric_v2:DriftedColumnsCount"
_VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"


def get_data_from_postgresql_db(db_uri: str, query: str) -> pd.DataFrame:
    """Read data from the postgresql database."""
    df = pl.read_database_uri(query=query, uri=db_uri, engine="adbc")
    return df.to_pandas()


def save_snapshot_html(snapshot: Any, html_path: Path) -> None:
    """
    Write the rendered Evidently report to disk.

    Args:
        snapshot: Snapshot returned by ``Report.run``.
        html_path (Path): Destination file. Parent directories are created.

    """
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Evidently >= 0.7 exposes Snapshot.save_html. Fall back to rendering the
    # string ourselves so a renamed helper cannot cost us the whole report.
    if hasattr(snapshot, "save_html"):
        snapshot.save_html(str(html_path))
    else:
        html_path.write_text(snapshot.get_html_str(), encoding="utf-8")


def create_and_forward_data_drift_report_to_prometheus(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    html_path: Path | None = None,
) -> bool:
    """
    Create and forward the evidently data drift report to prometheus.

    Args:
        reference_data (pd.DataFrame): Baseline the current data is compared against.
        current_data (pd.DataFrame): Recently served transactions.
        html_path (Path | None): When given, the rendered HTML report is written
            there so it can be served and inspected later.

    Returns:
        bool: True if the share of drifted columns reached the threshold.

    """
    try:
        # 1. Create and run report -> run() returns a Snapshot in Evidently >= 0.7
        report = Report([DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)])
        snapshot = report.run(current_data=current_data, reference_data=reference_data)

        # 2. Persist the HTML before parsing anything, so a changed metric layout
        # below still leaves a readable report behind for debugging.
        if html_path is not None:
            save_snapshot_html(snapshot, html_path)

        # 3. Results as a plain dict (no temp file needed)
        results = snapshot.dict()

        # 4. Dataset-level drift: DriftedColumnsCount -> {"count": ..., "share": ...}
        dataset_metric = None
        column_scores: dict[str, float] = {}

        for entry in results.get("metrics", []):
            metric_type = entry.get("config", {}).get("type")
            if metric_type == _DRIFTED_COUNT_TYPE:
                dataset_metric = entry
            elif metric_type == _VALUE_DRIFT_TYPE:
                column = entry["config"]["column"]
                column_scores[column] = float(entry["value"])

        if dataset_metric is None:
            raise KeyError("DriftedColumnsCount metric not found in report output.")

        value = dataset_metric["value"]
        drifted_count = float(value["count"])
        drifted_share = float(value["share"])
        threshold = dataset_metric["config"].get("drift_share", DRIFT_SHARE_THRESHOLD)
        drift_detected = drifted_share >= threshold

        # 5. Set Prometheus metrics
        DATA_DRIFT_DETECTED.set(int(drift_detected))
        DRIFTED_COLUMNS_RATIO.set(drifted_share)
        NUMBER_OF_DRIFTED_COLUMNS.set(drifted_count)

        for col_name, score in column_scores.items():
            FEATURE_DRIFT_SCORE.labels(feature_name=col_name).set(score)

        logger.info(
            "Evidently report finished: drift_detected=%s, drifted_share=%.3f",
            drift_detected,
            drifted_share,
        )

        return drift_detected

    except KeyError:
        # Callers must be able to tell a broken report apart from "no drift", so
        # every failure is re-raised: returning False here would silently suppress
        # retraining. The background caller in drift_report.py contains it.
        logger.exception("KeyError while parsing the Evidently report dictionary.")
        raise
    except Exception:
        logger.exception("Unexpected error during the Evidently drift calculation.")
        raise
