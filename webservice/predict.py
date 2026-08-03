"""Helpers for loading the registered MLflow model and serving predictions."""

import os
from functools import lru_cache

import mlflow
import pandas as pd
from data_model import TransactionKnownLabel, TransactionUnknownLabel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


# Cache the loaded model so repeated monitoring traffic does not reload the same
# MLflow artifact for every request.
@lru_cache(maxsize=1)
def load_model(model_name, alias="production"):
    """

    Load a registered MLflow model from the model registry.

    Args:
        model_name: Registered MLflow model name.
        alias: Model alias to resolve in MLflow. Defaults to "production".

    Returns:
        mlflow.pyfunc.PyFuncModel: Loaded MLflow model.

    """
    model_uri = f"models:/{model_name}@{alias}"

    # mlflow.pyfunc.load_model hides the concrete library flavor behind a common
    # prediction interface, so the API can stay the same even if the training
    # script later swaps the underlying estimator.
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(
    model_name,
    data: TransactionUnknownLabel | TransactionKnownLabel | pd.DataFrame,
    alias="production",
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
):
    """
    Predict a fraud label for the provided input data.

    Args:
        model_name: Registered MLflow model name.
        data: Input record containing the feature values to score.
        alias: Model alias to resolve in MLflow. Defaults to "production".
        mlflow_tracking_uri: MLflow tracking URI. Defaults to the
            ``MLFLOW_TRACKING_URI`` environment variable.

    Raises:
        RuntimeError: If ``MLFLOW_TRACKING_URI`` is not set.

    Returns:
        int: Predicted fraud label.

    """
    # Load .env when running locally outside Docker. In Docker Compose, the same
    # variable is injected through env_file.
    if not mlflow_tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set.")

    # MLflow needs the tracking URI before resolving models:/ URIs.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print("Data input:", data)

    # Convert the Pydantic request object into the tabular shape expected by
    # mlflow.pyfunc models.
    if isinstance(data, pd.DataFrame):
        df_transactions_input = data.copy()
    else:
        df_transactions_input = pd.DataFrame([data.model_dump()])
    print("Load model...")

    # Remove the target column
    for target_col in ["class", "Class", "target_class"]:
        if target_col in df_transactions_input.columns:
            df_transactions_input = df_transactions_input.drop(columns=[target_col])

    # Column mapping for the time, amount and the 'V' columns
    column_mapping = {"Time": "elapsed_sec", "Amount": "amount"}
    column_mapping.update({f"V{i}": f"pc_{i}" for i in range(1, 29)})

    # Rename the columns
    df_transactions_input = df_transactions_input.rename(columns=column_mapping)

    df_transactions_input = df_transactions_input.astype(float)

    # The cached loader keeps repeated monitoring traffic fast. Without it, the
    # API would reopen the same model artifact for every single request.
    model = load_model(model_name, alias)
    print("Making prediction with data: ", df_transactions_input.head())
    predictions = model.predict(df_transactions_input)

    results = [int(pred) for pred in predictions]

    if isinstance(data, (pd.DataFrame, list)):
        return results
    # Für ein einzelnes Pydantic-Objekt nur den einzelnen Integer zurückgeben
    return results[0]
