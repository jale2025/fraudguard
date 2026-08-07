from prometheus_client import Gauge

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
