"""Helpers for loading the registered MLflow model and serving predictions."""

import os
import threading

import mlflow
import pandas as pd
from data_model import TransactionKnownLabel, TransactionUnknownLabel
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str], tuple[str, PyFuncModel]] = {}


def _resolve_version(model_name: str, alias: str) -> str:
    """Resolve the alias to a concrete model version number."""
    try:
        return MlflowClient().get_model_version_by_alias(model_name, alias).version
    except MlflowException as exc:
        raise ModelNotAvailableError(
            f"Model '{model_name}' with alias '{alias}' is unavailable."
        ) from exc


class ModelNotAvailableError(RuntimeError):
    """Raised when the registered MLflow model cannot be loaded."""


def load_model(model_name: str, alias: str = "production"):
    """Return the model behind the alias, reloading it only when the version changed."""
    version = _resolve_version(model_name, alias)
    key = (model_name, alias)

    cached = _model_cache.get(key)
    if cached is not None and cached[0] == version:
        return cached[1]

    with _model_lock:
        # Re-check inside the lock: a concurrent request may have loaded it already.
        cached = _model_cache.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]

        try:
            # Load by explicit version, not by alias, so the alias cannot move
            # between resolution and load.
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
            print(
                f"Loaded model '{model_name}' v{version} (run_id={model.metadata.run_id})"
            )
        except MlflowException as exc:
            raise ModelNotAvailableError(
                f"Model '{model_name}' v{version} could not be loaded."
            ) from exc

        _model_cache[key] = (version, model)
        return model


def ensure_model_available(
    model_name: str,
    alias: str = "production",
    mlflow_tracking_uri: str | None = None,
):
    """Return the configured MLflow model if it is available."""
    tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ModelNotAvailableError("MLFLOW_TRACKING_URI is not configured.")

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


def served_model_info(model_name: str, alias: str = "production") -> dict:
    """Report what is actually loaded versus what the registry currently points at."""
    key = (model_name, alias)
    cached = _model_cache.get(key)

    if cached is None:
        return {"loaded": None, "registry": _resolve_version(model_name, alias)}

    loaded_version, model = cached

    return {
        "cached_version": loaded_version,
        "loaded_run_id": model.metadata.run_id,
        "loaded_model_uuid": model.metadata.model_uuid,
        "mlflow_registry_version": _resolve_version(model_name, alias),
        "worker_pid": os.getpid(),
    }
