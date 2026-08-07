import os

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

# Read environment variables
DB_URI = os.environ.get("DB_URI", "")

DRIFT_SHARE_THRESHOLD = 0.5

_DRIFTED_COUNT_TYPE = "evidently:metric_v2:DriftedColumnsCount"
_VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"


def get_data_from_postgresql_db(db_uri: str, query: str) -> pd.DataFrame:
    """Read data from the postgresql database."""
    df = pl.read_database_uri(query=query, uri=db_uri, engine="adbc")
    return df.to_pandas()


def create_and_forward_data_drift_report_to_prometheus(
    reference_data: pd.DataFrame, current_data: pd.DataFrame
) -> bool:
    """Create and forward the evidently data drift report to prometheus."""
    try:
        # 1. Create and run report -> run() returns a Snapshot in Evidently >= 0.7
        report = Report([DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)])
        snapshot = report.run(current_data=current_data, reference_data=reference_data)

        # 2. Results as a plain dict (no temp file needed)
        results = snapshot.dict()

        # 3. Dataset-level drift: DriftedColumnsCount -> {"count": ..., "share": ...}
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

        # 4. Set Prometheus metrics
        DATA_DRIFT_DETECTED.set(int(drift_detected))
        DRIFTED_COLUMNS_RATIO.set(drifted_share)
        NUMBER_OF_DRIFTED_COLUMNS.set(drifted_count)

        for col_name, score in column_scores.items():
            FEATURE_DRIFT_SCORE.labels(feature_name=col_name).set(score)

        print(f"DEBUG EVIDENTLY REPORT - DRIFT DETECTED: {drift_detected}")

        return drift_detected

    except KeyError as exc:
        print(f"KeyError while parsing Evidently report dictionary: {exc}")
        raise
    except Exception as exc:
        print(f"Unexpected error during Evidently drift calculation: {exc}")
        raise
