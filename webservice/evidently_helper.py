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
DB_URI = os.environ["DB_URI"]

def get_data_from_postgresql_db(db_uri: str, query: str) -> pl.DataFrame:
    """
    Read data from the postgresql database.

    Args:
        db_uri (str): Postgresql database uri
        query (str): SQL query

    Returns:
        pl.DataFrame: Get all data from the postgresql database.

    """
    df = pl.read_database_uri(
        query=query,
        uri=db_uri,
        engine="adbc",
    )

    return df.to_pandas()


def create_and_forward_data_drift_report_to_prometheus(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
    """
    Create and forward the evidently data drift report to prometheus.

    Args:
        reference_data (pd.DataFrame): Reference dataset
        current_data (pd.DataFrame): Current dataset

    Returns:
        bool: Drift detected or not

    """
    try:

        # Create the evidently data drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # Get the results as a dictionary
        results_dict = report.as_dict()

        # Extract the drift metrics of the result dict
        drift_metrics = results_dict['metrics'][0]["result"]

        # Check a data drift was detected
        drift_detected = 1 if drift_metrics["dataset_drift"] else 0

        # If so, set the gauge obj to 1, else to 0
        DATA_DRIFT_DETECTED.set(drift_detected)

        # Set the gauge values for th number of drifted columns and drifted columns ratio
        DRIFTED_COLUMNS_RATIO.set(drift_metrics["share_of_drifted_columns"])
        NUMBER_OF_DRIFTED_COLUMNS.set(drift_metrics["number_of_drifted_metrics"])

        # Set the drift scores for each feature column
        column_drift_dict = drift_metrics.get("drift_by_columns", {})

        for col_name, col_stats in column_drift_dict.items():
            score = col_stats.get("drift_score", 0.0)
            FEATURE_DRIFT_SCORE.labels(feature_name=col_name).set(score)

        return True if drift_detected == 1 else False

    except KeyError as exc:
        print(f"KeyError while parsing Evidently report dictionary: {exc}")
        raise exc
    except Exception as exc:
        print(f"Unexpected error during Evidently drift calculation: {exc}")
        raise exc
