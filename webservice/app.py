"""
FastAPI entrypoint for the local prediction service.

This module owns the public API surface:
- GET / for a simple info message
- GET /health for a liveness check
- POST /predict for model inference
- /metrics through prometheus-fastapi-instrumentator for service telemetry
"""
import io
import os
from typing import Any

import pandas as pd
from data_model import (
    TransactionClassificationKnownLabel,
    TransactionClassificationUnknownLabel,
    TransactionKnownLabel,
    TransactionUnknownLabel,
)
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from predict import predict

# Load dotenv
load_dotenv()

# Load environment variables from the composed environment.
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = os.environ["MODEL_NAME"]
DEFAULT_MODEL_ALIAS = os.environ["DEFAULT_MODEL_ALIAS"]
# MONITORING_URL = os.environ["MONITORING_URL"]

app = FastAPI(title="Credit Card Fraud Detection API", version="0.1")

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


@app.post("/predict_single_unknown_label", response_model=TransactionClassificationUnknownLabel)
def predict_transaction_unknown_label(data: TransactionUnknownLabel) -> dict[str, Any]:
    """Run model inference on a transaction and return the classification."""
    # First serve the model prediction. Monitoring should observe this request,
    # but it should not change the prediction result returned to the client.
    prediction = predict(REGISTERED_MODEL_NAME, data, DEFAULT_MODEL_ALIAS)
    # try:
    #     print(f"Sending data to metrics application: {data}")

    #     # Evidently needs both the input features and the model output so it can
    #     # compare live predictions against the reference distribution.
    #     monitoring_payload = TransactionClassification(
    #         **data.model_dump(), prediction=prediction
    #     ).model_dump()

    #     # This POST is intentionally fire-and-forget from the API's point of
    #     # view. If monitoring is slow or unavailable, the client should still
    #     # receive the prediction response from the model service.
    #     requests.post(
    #         MONITORING_URL,
    #         json=monitoring_payload,
    #         timeout=5,
    #     )
    # except requests.exceptions.ConnectionError as error:
    #     print(f"Cannot reach a metrics application, error: {error}, data: {data}")
    # except requests.exceptions.Timeout as error:
    #     print(f"Metrics application timed out, error: {error}, data: {data}")

    # Return the same payload shape used for monitoring so users can
    # compare what the client sees with what Evidently receives.
    print(f"Returning prediction: {prediction}")
    return TransactionClassificationUnknownLabel(**data.model_dump(), prediction=prediction)


@app.post("/predict_single_known_label", response_model=TransactionClassificationKnownLabel)
def predict_transaction_known_label(data: TransactionKnownLabel) -> dict[str, Any]:
    """Run model inference on a transaction and return the classification."""
    # First serve the model prediction. Monitoring should observe this request,
    # but it should not change the prediction result returned to the client.
    prediction = predict(REGISTERED_MODEL_NAME, data, DEFAULT_MODEL_ALIAS)

    # Return the same payload shape used for monitoring so users can
    # compare what the client sees with what Evidently receives.
    print(f"Returning prediction: {prediction}")
    return TransactionClassificationKnownLabel(**data.model_dump(), prediction=prediction)

@app.post("/predict_file_unknown_label", response_model=list[TransactionClassificationUnknownLabel])
async def predict_transactions_unknown_label(file: UploadFile = File(description="Parquet file containing the transactions with unknown labels.")) -> list[dict[str, Any]]:
    """Run model inference on a uploaded Parquet file without ground truth."""
    if not file.filename.endswith(".parquet"):
            raise HTTPException(status_code=400, detail="Only .parquet files are supported.")
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
        raise HTTPException(status_code=500, detail=f"Error processing Parquet file: {str(e)}")


@app.post("/predict_file_known_label", response_model=list[TransactionClassificationKnownLabel])
async def predict_transactions_known_label(
    file: UploadFile = File(description="Parquet file containing the transactions with known labels.")
) -> list[dict[str, Any]]:
    if not file.filename.endswith(".parquet"):
        raise HTTPException(status_code=400, detail="Only .parquet files are supported.")
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
        raise HTTPException(status_code=500, detail=f"Error processing Parquet file: {str(e)}")
