"""
Integration tests for the prediction endpoints.

These drive the real ASGI stack -- routing, pydantic validation and response
serialisation -- and only replace the model, the database write and the drift report.
"""

import io
from typing import Any

from predict import ModelNotAvailableError


def test_predict_single_unknown_label_returns_prediction(
    client: Any, stubs: Any, transaction: dict[str, float]
) -> None:
    """Verify the response echoes the request and the row is queued for the database."""
    response = client.post("/predict_single_unknown_label", json=transaction)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert {key: body[key] for key in transaction} == transaction

    assert stubs.forward_calls[0]["table_name"] == "predictions"
    assert stubs.report_calls[0]["counter_name"] == "not_labeled_queue"


def test_predict_single_known_label_returns_class(
    client: Any, stubs: Any, transaction: dict[str, float]
) -> None:
    """Verify a labelled transaction keeps its ``class`` and lands in the label queue."""
    transaction["class"] = 0

    response = client.post("/predict_single_known_label", json=transaction)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    # The response model serialises by alias, so clients get "class" back, not
    # the internal target_class field name.
    assert body["class"] == 0
    assert "target_class" not in body

    assert stubs.forward_calls[0]["table_name"] == "labeled_predictions_queue"
    assert stubs.report_calls[0]["counter_name"] == "labeled_queue"


def test_predict_rejects_incomplete_transaction(
    client: Any, stubs: Any, transaction: dict[str, float]
) -> None:
    """Verify a missing feature is rejected by validation before the model is called."""
    del transaction["pc_7"]

    response = client.post("/predict_single_unknown_label", json=transaction)

    assert response.status_code == 422
    assert any(detail["loc"][-1] == "pc_7" for detail in response.json()["detail"])
    assert stubs.predict_calls == []


def test_predict_returns_503_when_model_unavailable(
    client: Any, stubs: Any, transaction: dict[str, float]
) -> None:
    """Verify an unavailable registry surfaces as 503 rather than a generic 500."""
    stubs.error = ModelNotAvailableError("registry is down")

    response = client.post("/predict_single_unknown_label", json=transaction)

    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction model is currently unavailable."
    assert stubs.forward_calls == []


def test_predict_file_maps_raw_column_names(
    client: Any, stubs: Any, raw_transactions: Any
) -> None:
    """Verify an uploaded parquet is renamed to the canonical feature names."""
    buffer = io.BytesIO()
    raw_transactions(rows=3).to_parquet(buffer, index=False)

    response = client.post(
        "/predict_file_unknown_label",
        files={
            "file": (
                "transactions.parquet",
                buffer.getvalue(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 200
    assert len(response.json()) == 3

    scored = stubs.predict_calls[0]["data"]
    assert "Time" not in scored.columns
    assert {"elapsed_sec", "amount", "pc_1", "pc_28"} <= set(scored.columns)

    # A non-parquet upload is refused before anything is read.
    rejected = client.post(
        "/predict_file_unknown_label",
        files={"file": ("transactions.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "Only .parquet files are supported."
