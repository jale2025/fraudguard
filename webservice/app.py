"""
FastAPI entrypoint for the local prediction service.

This module owns the public API surface:
- GET / for a simple info message
- GET /demo for the presentation interface
- GET /ready for model readiness
- GET /model_info for the served model version
- POST /predict_* for model inference
- /drift_report/* for the Evidently HTML reports
- /metrics for prometheus monitoring
"""

import io
import os
from pathlib import Path
from typing import Annotated, Literal

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
from drift_report import (
    QUEUE_QUERIES,
    claim_report_slot,
    create_report_and_trigger_workflow,
    latest_report_file,
    list_reports,
    report_file,
    run_drift_report,
)
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from predict import (
    ModelNotAvailableError,
    ensure_model_available,
    predict,
    served_model_info,
)
from prometheus_client import make_asgi_app

QueueName = Literal["labeled_queue", "not_labeled_queue"]
DEMO_PAGE = Path(__file__).resolve().parent / "frontend" / "demo.html"

# Load dotenv
load_dotenv()

# Load environment variables from the composed environment.
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = os.environ["MODEL_NAME"]
DEFAULT_MODEL_ALIAS = os.environ["DEFAULT_MODEL_ALIAS"]
# Not used here anymore -- api_to_database and drift_report read it themselves -- but
# both read it lazily, so this keeps a missing DB_URI a startup failure.
DB_URI = os.environ["DB_URI"]

app = FastAPI(title="Credit Card Fraud Detection API", version="0.1")

# Expose Prometheus metrics on /metrics for Prometheus to scrape.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def index():
    """Verify that the API is alive."""
    return {"message": "Credit Card Fraud Detection API"}


@app.get("/demo", include_in_schema=False, response_class=FileResponse)
def demo() -> FileResponse:
    """Serve the lightweight presentation interface."""
    return FileResponse(DEMO_PAGE)


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


@app.get("/model_info")
def model_info() -> dict:
    """Report which model version this worker is currently serving."""
    try:
        return served_model_info(REGISTERED_MODEL_NAME, DEFAULT_MODEL_ALIAS)
    except ModelNotAvailableError as exc:
        # The registry may be down while a model is still loaded and serving.
        # Report what we have instead of failing the diagnostic call.
        return {"registry_error": str(exc), "worker_pid": os.getpid()}


@app.post(
    "/predict_single_unknown_label",
    response_model=TransactionClassificationUnknownLabel,
)
def predict_transaction_unknown_label(
    data: TransactionUnknownLabel,
    background_tasks: BackgroundTasks,
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

        print(f"Returning prediction: {prediction}")

        response = TransactionClassificationUnknownLabel(
            **data.model_dump(), prediction=prediction
        )

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="predictions", data=response)

        # Only the redis counter check happens here; the report itself runs after
        # the response was sent. The task is attached to the returned response, so a
        # handler that raises below never fires a report.
        create_report_and_trigger_workflow(
            counter_name="not_labeled_queue",
            current_data_query=QUEUE_QUERIES["not_labeled_queue"],
            background_tasks=background_tasks,
        )

        # Count the request only once the whole handler succeeded. Counting earlier
        # would also let a later failure add an "internal_error" increment, so a
        # single request would be counted twice and reported as a success.
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        return response

    except HTTPException:
        # Do not turn a deliberate 4xx into a generic 500 further down.
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


@app.post(
    "/predict_single_known_label", response_model=TransactionClassificationKnownLabel
)
def predict_transaction_known_label(
    data: TransactionKnownLabel,
    background_tasks: BackgroundTasks,
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

        print(f"Returning prediction: {prediction}")

        response = TransactionClassificationKnownLabel(
            **data.model_dump(), prediction=prediction
        )

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="labeled_predictions_queue", data=response)

        # Only the redis counter check happens here; the report itself runs after
        # the response was sent.
        create_report_and_trigger_workflow(
            counter_name="labeled_queue",
            current_data_query=QUEUE_QUERIES["labeled_queue"],
            background_tasks=background_tasks,
        )

        # Count the request only once the whole handler succeeded. Counting earlier
        # would also let a later failure add an "internal_error" increment, so a
        # single request would be counted twice and reported as a success.
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        return response

    except HTTPException:
        # Do not turn a deliberate 4xx into a generic 500 further down.
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
    background_tasks: BackgroundTasks,
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

    # Check if a class column exists (in any casing), if so throw an exception
    if has_class_column(df):
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="invalid_input",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail="The uploaded Parquet file for unknown labels must NOT contain a 'class' or 'Class' column.",
        )

    try:
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

        # Only the redis counter check happens here. This endpoint is async, so
        # running the report inline would block the event loop of the whole worker.
        create_report_and_trigger_workflow(
            counter_name="not_labeled_queue",
            current_data_query=QUEUE_QUERIES["not_labeled_queue"],
            background_tasks=background_tasks,
        )

        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="success",
        ).inc()

        # Return the df as a list of dictionaries
        return [
            TransactionClassificationUnknownLabel.model_validate(record)
            for record in df.to_dict(orient="records")
        ]

    except HTTPException:
        # Do not turn a deliberate 4xx into a generic 500 further down.
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


@app.post(
    "/predict_file_known_label",
    response_model=list[TransactionClassificationKnownLabel],
)
async def predict_transactions_known_label(
    file: Annotated[
        UploadFile,
        File(description="Parquet file containing the transactions with known labels."),
    ],
    background_tasks: BackgroundTasks,
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

    # Check if a class column exists (in any casing), if not throw an exception
    if not has_class_column(df):
        PREDICTION_REQUESTS.labels(
            endpoint=endpoint,
            result="invalid_input",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail="The uploaded Parquet file does not have a 'class' or 'Class' column.",
        )

    try:
        # Column mapping for the time, amount and the 'V' columns
        column_mapping = {"Time": "elapsed_sec", "Amount": "amount"}
        column_mapping.update({f"V{i}": f"pc_{i}" for i in range(1, 29)})

        # Column mapping for the 'class' mapping
        if "Class" in df.columns:
            column_mapping["Class"] = "class"

        # Rename the columns
        df = df.rename(columns=column_mapping)

        actual_labels = df["class"].astype(int).tolist()

        with MODEL_INFERENCE_DURATION.labels(
            request_type="file",
        ).time():
            predictions = predict(
                REGISTERED_MODEL_NAME,
                df,
                DEFAULT_MODEL_ALIAS,
            )

        # Record the model metrics directly after inference, mirroring the
        # unknown-label endpoint. Recording them after the database write would
        # drop them whenever the persistence or Evidently path fails, even though
        # the inference itself succeeded.
        record_prediction_metrics(
            predictions,
            request_type="file",
            ground_truth="known",
            actual=actual_labels,
        )

        df["prediction"] = predictions

        # Call the function to forward the incoming transactions into database
        forward_to_database(table_name="labeled_predictions_queue", data=df)

        # Only the redis counter check happens here. This endpoint is async, so
        # running the report inline would block the event loop of the whole worker.
        create_report_and_trigger_workflow(
            counter_name="labeled_queue",
            current_data_query=QUEUE_QUERIES["labeled_queue"],
            background_tasks=background_tasks,
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


@app.post("/drift_report/trigger", status_code=202)
def trigger_drift_report(
    background_tasks: BackgroundTasks,
    queue: QueueName = "labeled_queue",
    trigger_retraining: bool = False,
) -> dict[str, str]:
    """
    Run a drift report on demand, ignoring the row counter.

    Args:
        background_tasks (BackgroundTasks): Collector for the work that runs once
            the response has been sent.
        queue (QueueName): Prediction queue to compare against the training data.
        trigger_retraining (bool): Whether detected drift may start the retraining
            deployment. Defaults to False so a debug run cannot retrain by accident.

    Returns:
        dict[str, str]: The scheduled queue.

    """
    if not claim_report_slot():
        raise HTTPException(
            status_code=409,
            detail="A drift report is already running.",
        )

    background_tasks.add_task(
        run_drift_report,
        counter_name=queue,
        current_data_query=QUEUE_QUERIES[queue],
        trigger_retraining=trigger_retraining,
    )

    return {"status": "scheduled", "queue": queue}


@app.get("/drift_report/latest", response_class=FileResponse)
def latest_drift_report(queue: QueueName = "labeled_queue") -> FileResponse:
    """Serve the most recent Evidently HTML drift report for a queue."""
    path = latest_report_file(queue)

    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No drift report has been generated yet for '{queue}'.",
        )

    return FileResponse(path, media_type="text/html")


@app.get("/drift_report/history")
def drift_report_history(queue: QueueName | None = None) -> list[dict]:
    """List the retained Evidently drift reports, newest first."""
    return list_reports(queue)


@app.get("/drift_report/file/{filename}", response_class=FileResponse)
def drift_report_by_name(filename: str) -> FileResponse:
    """Serve one retained report by name, as listed by /drift_report/history."""
    # report_file resolves inside the reports directory and returns None for a
    # traversal attempt, so an unexpected name is a 404 rather than a file leak.
    path = report_file(filename)

    if path is None:
        raise HTTPException(status_code=404, detail="Unknown drift report.")

    return FileResponse(path, media_type="text/html")


def has_class_column(df: pd.DataFrame) -> bool:
    """
    Check case-insensitively whether a ground truth class column is present.

    The raw Kaggle dataset ships the label as 'Class', while the database
    schema expects 'class'. Comparing the normalized name avoids missing the
    column just because of its casing or surrounding whitespace.

    Args:
        df (pd.DataFrame): DataFrame read from the uploaded Parquet file.

    Returns:
        bool: True if a class column exists in any casing, otherwise False.

    """
    return any(str(col).strip().lower() == "class" for col in df.columns)
