from prometheus_client import Counter, Gauge, Histogram

# Overall dataset drift status (1 = drift detected, 0 = no drift)
DATA_DRIFT_DETECTED = Gauge(
    "fraudguard_data_drift_detected",
    "1 if overall dataset drift is detected, 0 otherwise",
)

# Share of drifted columns/features (0.0 to 1.0)
DRIFTED_COLUMNS_RATIO = Gauge(
    "fraudguard_drifted_columns_ratio",
    "Ratio of drifted features in the dataset",
)

# Total number of drifted columns/features
NUMBER_OF_DRIFTED_COLUMNS = Gauge(
    "fraudguard_number_of_drifted_columns",
    "Number of drifted features",
)

# Drift score per individual feature (e.g., p-value or Wasserstein distance)
FEATURE_DRIFT_SCORE = Gauge(
    "fraudguard_feature_drift_score",
    "Drift score for a specific feature",
    ["feature_name"],
)

# The report runs in the background, after the response was already sent, so a
# failure can no longer surface as a non-2xx status code. These metrics are the
# only signal that the drift monitoring itself is still healthy.
DRIFT_REPORT_RUNS = Counter(
    "fraudguard_drift_report_runs_total",
    "Outcomes of drift-report runs scheduled by the prediction endpoints.",
    ["queue", "result"],
)

DRIFT_REPORT_DURATION = Histogram(
    "fraudguard_drift_report_duration_seconds",
    "Wall-clock time of a drift report, including both database reads.",
    ["queue"],
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

DRIFT_REPORT_IN_PROGRESS = Gauge(
    "fraudguard_drift_report_in_progress",
    "1 while a drift report is running in the background, 0 otherwise",
)

LAST_DRIFT_REPORT_TIMESTAMP = Gauge(
    "fraudguard_last_drift_report_timestamp_seconds",
    "Unix timestamp of the last successfully completed drift report.",
    ["queue"],
)

# The redis counters the prediction endpoints bump, one per prediction queue.
QUEUES = ("labeled_queue", "not_labeled_queue")

# "skipped_busy" is only recorded when the row threshold *was* reached but a report
# was already running. A below-threshold request schedules nothing at all and is a
# no-op, not a skip -- counting it would just re-measure request volume.
# "counter_unavailable" means the redis counter could not even be read.
DRIFT_REPORT_RESULTS = ("success", "failure", "skipped_busy", "counter_unavailable")


def initialise_children() -> None:
    """
    Export every reachable drift-report child at zero from process start.

    A labelled child does not exist in the registry until ``.labels()`` is called
    for it, so it is first scraped already holding 1 and range queries such as
    ``increase()`` and ``rate()`` silently lose that first observation. Creating
    the children up front gives those queries the zero baseline they need.
    """
    for queue in QUEUES:
        DRIFT_REPORT_DURATION.labels(queue=queue)

        for result in DRIFT_REPORT_RESULTS:
            DRIFT_REPORT_RUNS.labels(queue=queue, result=result)


initialise_children()
DRIFT_REPORT_IN_PROGRESS.set(0)
