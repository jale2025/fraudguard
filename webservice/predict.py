"""Helpers for loading the registered MLflow model and serving predictions."""

from functools import lru_cache
import mlflow
import os

from dotenv import load_dotenv
import pandas as pd


# Cache the loaded model so repeated monitoring traffic does not reload the same
# MLflow artifact for every request.
@lru_cache(maxsize=1)
def load_model(model_name, alias="production"):
    model_uri = f"models:/{model_name}@{alias}"

    # mlflow.pyfunc.load_model hides the concrete library flavor behind a common
    # prediction interface, so the API can stay the same even if the training
    # script later swaps the underlying estimator.
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model_name, data, alias="production"):
    # Load .env when running locally outside Docker. In Docker Compose, the same
    # variable is injected through env_file.
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set.")

    # MLflow needs the tracking URI before resolving models:/ URIs.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Data input:", data)

    # Convert the Pydantic request object into the tabular shape expected by
    # mlflow.pyfunc models.
    model_input = pd.DataFrame([data.model_dump()]).astype(float)
    print("Load model...")

    # The cached loader keeps repeated monitoring traffic fast. Without it, the
    # API would reopen the same model artifact for every single request.
    model = load_model(model_name, alias)
    print("Making prediction with data: ", model_input.head())
    prediction = model.predict(model_input)
    return int(prediction[0])
