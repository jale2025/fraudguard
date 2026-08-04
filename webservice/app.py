"""
FastAPI entrypoint for the local prediction service.

This module owns the public API surface:
- GET / for a simple info message
- GET /health for a liveness check
- POST /predict for model inference
- /metrics for prometheus monitoring
"""

import io
import os
from typing import Annotated, Any

import pandas as pd
from data_model import (
    TransactionClassificationKnownLabel,
    TransactionClassificationUnknownLabel,
    TransactionKnownLabel,
    TransactionUnknownLabel,
)
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from predict import predict, ModelNotAvailableError
from prometheus_client import make_asgi_app
from api_model_metrics import MODEL_INFERENCE_DURATION, record_prediction_metrics
from api_http_metrics import PREDICTION_REQUESTS

# Load dotenv
load_dotenv()

# Load environment variables from the composed environment.
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = os.environ["MODEL_NAME"]
DEFAULT_MODEL_ALIAS = os.environ["DEFAULT_MODEL_ALIAS"]

app = FastAPI(title="Credit Card Fraud Detection API", version="0.1")

# Expose Prometheus metrics on /metrics for Prometheus to scrape.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Expose default FastAPI request metrics on /metrics for Prometheus.
# Instrumentator().instrument(app).expose(app)
@app.get("/")
def index():
    """Verify that the API is alive."""
    return {"message": "Credit Card Fraud Detection API"}


@app.get("/health")
def health() -> dict[str, str]:
    """Check the API liveness."""
    return {"status": "ok"}


@app.post(
    "/predict_single_unknown_label",
    response_model=TransactionClassificationUnknownLabel,
)
def predict_transaction_unknown_label(data: TransactionUnknownLabel) -> TransactionClassificationUnknownLabel:
    """Run model inference on a transaction and return the classification."""
    # First serve the model prediction. Monitoring should observe this request,
    # but it should not change the prediction result returned to the client.
    # prediction = predict(REGISTERED_MODEL_NAME, data, DEFAULT_MODEL_ALIAS)
    endpoint = "predict_single_unknown_label"
    try:
        with MODEL_INFERENCE_DURATION.labels(
            request_type="single",
        ).time():
            prediction = predict(
                REGISTERED_MODEL_NAME,
                data,
                DEFAULT_MODEL_ALIAS,
            )

        record_prediction_metrics(
            prediction,
            request_type="single",
            ground_truth="unknown",
        )

        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        print(f"Returning prediction: {prediction}")

        return TransactionClassificationUnknownLabel(
            **data.model_dump(), prediction=prediction
        )

    except ModelNotAvailableError as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="model_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc


@app.post(
    "/predict_single_known_label", response_model=TransactionClassificationKnownLabel
)
def predict_transaction_known_label(data: TransactionKnownLabel) -> TransactionClassificationKnownLabel:
    """Run model inference on a transaction and return the classification."""

    endpoint = "predict_single_known_label"
    try:
        with MODEL_INFERENCE_DURATION.labels(
            request_type="single",
        ).time():
            prediction = predict(
                REGISTERED_MODEL_NAME,
                data,
                DEFAULT_MODEL_ALIAS,
            )

        record_prediction_metrics(
            prediction,
            request_type="single",
            ground_truth="known",
        )

        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        print(f"Returning prediction: {prediction}")

        return TransactionClassificationKnownLabel(
            **data.model_dump(), prediction=prediction
        )

    except ModelNotAvailableError as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="model_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc


@app.post(
    "/predict_file_unknown_label",
    response_model=list[TransactionClassificationUnknownLabel],
)
async def predict_transactions_unknown_label(
    file: Annotated[
        UploadFile,
        File(
            description="Parquet file containing the transactions with unknown labels."
        ),
    ],
) -> list[dict[str, Any]]:
    """Run model inference on a uploaded Parquet file without ground truth."""
    if not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=400, detail="Only .parquet files are supported."
        )
    try:
        # Reading the file byte stream asynchronous from the ram and store it in the variable
        contents = await file.read()

        # Transfer the byte array in an object and transform it in a df
        df = pd.read_parquet(io.BytesIO(contents))

        # Trim the df due to tiem reasons
        df = df.iloc[:500].copy()

        # Column mapping for the time, amount and the 'V' columns
        column_mapping = {"Time": "elapsed_sec", "Amount": "amount"}
        column_mapping.update({f"V{i}": f"pc_{i}" for i in range(1, 29)})

        # Rename the columns
        df = df.rename(columns=column_mapping)

        # Apply the predict function on the df
        predictions = predict(REGISTERED_MODEL_NAME, df, DEFAULT_MODEL_ALIAS)

        # Add another column for the predictions
        df["prediction"] = predictions

        # Return the df as a list of dictionaries
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing Parquet file: {str(e)}"
        ) from e


@app.post(
    "/predict_file_known_label",
    response_model=list[TransactionClassificationKnownLabel],
)
async def predict_transactions_known_label(
    file: Annotated[
        UploadFile,
        File(description="Parquet file containing the transactions with known labels."),
    ],
) -> list[dict[str, Any]]:
    """Run model inference on an uploaded Parquet file with known labels."""
    if not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=400, detail="Only .parquet files are supported."
        )
    try:
        # Reading the file byte stream asynchronous from the ram and store it in the variable
        contents = await file.read()

        # Transfer the byte array in an object and transform it in a df
        df = pd.read_parquet(io.BytesIO(contents))

        # Trim the df due to tiem reasons
        df = df.iloc[:500].copy()

        # # Column mapping for the time, amount and the 'V' columns
        column_mapping = {"Time": "elapsed_sec", "Amount": "amount"}
        column_mapping.update({f"V{i}": f"pc_{i}" for i in range(1, 29)})

        # Column mapping for the 'class' mapping
        if "Class" in df.columns:
            column_mapping["Class"] = "class"

        # Rename the columns
        df = df.rename(columns=column_mapping)

        # Extract the 'class' column
        df_class_col = df["class"]

        # Apply the predict function on the df
        predictions = predict(REGISTERED_MODEL_NAME, df, DEFAULT_MODEL_ALIAS)

        # Add another column for the predictions
        df["prediction"] = predictions

        # Add the 'class' column to the df after extracting earlier
        df["class"] = df_class_col

        # Return the df as a list of dictionaries
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing Parquet file: {str(e)}"
        ) from e
