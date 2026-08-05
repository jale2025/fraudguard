"""Helpers for loading the registered MLflow model and serving predictions."""

import os
from functools import lru_cache

import mlflow
import pandas as pd
from data_model import TransactionKnownLabel, TransactionUnknownLabel
from mlflow.exceptions import MlflowException

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


class ModelNotAvailableError(RuntimeError):
    """Raised when the registered MLflow model cannot be loaded."""


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
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except MlflowException as exc:
        raise ModelNotAvailableError(
            f"Model '{model_name}' with alias '{alias}' is unavailable."
        ) from exc

def ensure_model_available(
    model_name: str,
    alias: str = "production",
    mlflow_tracking_uri: str | None = None,
):
    """Return the configured MLflow model if it is available."""

    tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ModelNotAvailableError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    mlflow.set_tracking_uri(tracking_uri)

    return load_model(model_name, alias)

def predict(
    model_name,
    data: TransactionUnknownLabel | TransactionKnownLabel | pd.DataFrame,
    alias="production",
    mlflow_tracking_uri: str | None = None,
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
    # # Ensure tracking URI is available
    # if not mlflow_tracking_uri:
    #     raise RuntimeError("MLFLOW_TRACKING_URI is not set.")

    # # Set MLflow tracking URI before resolving model reference
    # mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Convert Pydantic request object into a DataFrame if necessary
    if isinstance(data, pd.DataFrame):
        df_transactions_input = data.copy()
    else:
        df_transactions_input = pd.DataFrame([data.model_dump()])

    # Remove target ground-truth columns if present in the input
    for target_col in ["class", "Class", "target_class"]:
        if target_col in df_transactions_input.columns:
            df_transactions_input = df_transactions_input.drop(columns=[target_col])

    # Ensure all feature columns are converted to float
    df_transactions_input = df_transactions_input.astype(float)

    # Load model from registry using cached helper function
    # model = load_model(model_name, alias)
    model = ensure_model_available(
        model_name,
        alias,
        mlflow_tracking_uri,
    )
    # Perform inference
    predictions = model.predict(df_transactions_input)

    # Vectorized type conversion to integer using C-level NumPy/Pandas ops
    if hasattr(predictions, "astype"):
        results = predictions.astype(int)
    else:
        results = predictions

    # Return a list of ints for DataFrame batch requests, or a single scalar int for single records
    if isinstance(data, pd.DataFrame):
        return results.tolist()

    return int(results[0])
