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
from typing import Annotated

import pandas as pd
from api_http_metrics import PREDICTION_REQUESTS
from api_model_metrics import (
    MODEL_INFERENCE_DURATION,
    MODEL_READY,
    record_prediction_metrics,
)
from api_to_database import forward_to_database
from data_model import (
    TransactionClassificationKnownLabel,
    TransactionClassificationUnknownLabel,
    TransactionKnownLabel,
    TransactionUnknownLabel,
)
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from predict import ModelNotAvailableError, ensure_model_available, predict
from prometheus_client import make_asgi_app

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


@app.get("/")
def index():
    """Verify that the API is alive."""
    return {"message": "Credit Card Fraud Detection API"}


# # not used so far
# @app.get("/health")
# def health() -> dict[str, str]:
#     """Check the API liveness."""
#     return {"status": "ok"}


@app.get("/ready")
def readiness() -> dict[str, str]:
    """Check whether the prediction model is available."""
    try:
        ensure_model_available(
            REGISTERED_MODEL_NAME,
            DEFAULT_MODEL_ALIAS,
        )
    except ModelNotAvailableError as exc:
        MODEL_READY.set(0)

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc

    MODEL_READY.set(1)
    return {"status": "ready"}


@app.post(
    "/predict_single_unknown_label",
    response_model=TransactionClassificationUnknownLabel,
)
def predict_transaction_unknown_label(
    data: TransactionUnknownLabel,
) -> TransactionClassificationUnknownLabel:
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

        response = TransactionClassificationUnknownLabel(
            **data.model_dump(), prediction=prediction
        )

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="predictions", data=response)

        return response

    except ModelNotAvailableError as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="model_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc

    except Exception as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="internal_error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        ) from exc


@app.post(
    "/predict_single_known_label", response_model=TransactionClassificationKnownLabel
)
def predict_transaction_known_label(
    data: TransactionKnownLabel,
) -> TransactionClassificationKnownLabel:
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
            actual=data.target_class,
        )

        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        print(f"Returning prediction: {prediction}")

        response = TransactionClassificationKnownLabel(
            **data.model_dump(), prediction=prediction
        )

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="labeled_predictions_queue", data=response)

        return response

    except ModelNotAvailableError as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="model_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc

    except Exception as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="internal_error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
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
) -> list[TransactionClassificationUnknownLabel]:
    """Run model inference on a uploaded Parquet file without ground truth."""
    endpoint = "predict_file_unknown_label"

    if not file.filename or not file.filename.lower().endswith(".parquet"):
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="invalid_input",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail="Only .parquet files are supported.",
        )

    try:
        # Reading the file byte stream asynchronous from the ram and store it in the variable
        contents = await file.read()

        # Transfer the byte array in an object and transform it in a df
        df = pd.read_parquet(io.BytesIO(contents))

    except Exception as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="invalid_input",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid Parquet file.",
        ) from exc

    try:
        # Trim the df due to time limitation
        df = df.iloc[:500].copy()

        # Column mapping for the time, amount and the 'V' columns
        column_mapping = {"Time": "elapsed_sec", "Amount": "amount"}
        column_mapping.update({f"V{i}": f"pc_{i}" for i in range(1, 29)})

        # Rename the columns
        df = df.rename(columns=column_mapping)

        with MODEL_INFERENCE_DURATION.labels(
            request_type="file",
        ).time():
            predictions = predict(
                REGISTERED_MODEL_NAME,
                df,
                DEFAULT_MODEL_ALIAS,
            )

        record_prediction_metrics(
            predictions,
            request_type="file",
            ground_truth="unknown",
        )

        # Add another column for the predictions
        df["prediction"] = predictions

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="predictions", data=df)

        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        # Return the df as a list of dictionaries
        return [
            TransactionClassificationUnknownLabel.model_validate(record)
            for record in df.to_dict(orient="records")
        ]

    except ModelNotAvailableError as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="model_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc

    except Exception as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="internal_error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        ) from exc


@app.post(
    "/predict_file_known_label",
    response_model=list[TransactionClassificationKnownLabel],
)
async def predict_transactions_known_label(
    file: Annotated[
        UploadFile,
        File(description="Parquet file containing the transactions with known labels."),
    ],
) -> list[TransactionClassificationKnownLabel]:
    """Run model inference on an uploaded Parquet file with known labels."""
    endpoint = "predict_file_known_label"

    if not file.filename or not file.filename.lower().endswith(".parquet"):
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="invalid_input",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail="Only .parquet files are supported.",
        )

    try:
        # Reading the file byte stream asynchronous from the ram and store it in the variable
        contents = await file.read()

        # Transfer the byte array in an object and transform it in a df
        df = pd.read_parquet(io.BytesIO(contents))

    except Exception as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="invalid_input",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid Parquet file.",
        ) from exc

    try:
        # Trim the df due to time limitations
        df = df.iloc[:500].copy()

        # # Column mapping for the time, amount and the 'V' columns
        column_mapping = {"Time": "elapsed_sec", "Amount": "amount"}
        column_mapping.update({f"V{i}": f"pc_{i}" for i in range(1, 29)})

        # Column mapping for the 'class' mapping
        if "Class" in df.columns:
            column_mapping["Class"] = "class"

        # Rename the columns
        df = df.rename(columns=column_mapping)

        if "class" not in df.columns:
            PREDICTION_REQUESTS.labels(
                endpoint=endpoint,
                result="invalid_input",
            ).inc()

            raise HTTPException(
                status_code=422,
                detail="The Parquet file must contain a 'Class' column.",
            )

        actual_labels = df["class"].astype(int).tolist()

        with MODEL_INFERENCE_DURATION.labels(
            request_type="file",
        ).time():
            predictions = predict(
                REGISTERED_MODEL_NAME,
                df,
                DEFAULT_MODEL_ALIAS,
            )

        df["prediction"] = predictions

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="labeled_predictions_queue", data=df)

        record_prediction_metrics(
            predictions,
            request_type="file",
            ground_truth="known",
            actual=actual_labels,
        )

        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        # Return the df as a list of dictionaries
        return [
            TransactionClassificationKnownLabel.model_validate(record)
            for record in df.to_dict(orient="records")
        ]

    except HTTPException:
        raise

    except ModelNotAvailableError as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="model_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable.",
        ) from exc

    except Exception as exc:
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="internal_error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        ) from exc
