"""FastAPI entrypoint for the local prediction service.

This module owns the public API surface:
- GET / for a simple info message
- GET /health for a liveness check
- POST /predict for model inference 
- /metrics through prometheus-fastapi-instrumentator for service telemetry
"""

import os
from typing import Any

import requests
from data_model import Transaction, TransactionClassification
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from predict import predict
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

# Load environment variables from the composed environment.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = os.environ["REGISTERED_MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]
MONITORING_URL = os.environ["MONITORING_URL"]

app = FastAPI(title="Credit Card Fraud Detection API", version="0.1")

# Expose default FastAPI request metrics on /metrics for Prometheus.
# Instrumentator().instrument(app).expose(app)

@app.get("/")
def index():
    # Simple root route to verify the API is alive.
    return {"message": "Credit Card Fraud Detection API"}

@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness check for the API."""
    return {"status": "ok"}

@app.post("/predict", response_model=TransactionClassification)
def predict_transaction(data: Transaction) -> dict[str, Any]:
    # First serve the model prediction. Monitoring should observe this request,
    # but it should not change the prediction result returned to the client.
    prediction = predict(REGISTERED_MODEL_NAME, data, MODEL_ALIAS)
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
    return TransactionClassification(**data.model_dump(), prediction=prediction)
