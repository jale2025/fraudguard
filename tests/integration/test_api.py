"""
Integration tests for the prediction endpoints.

These drive the real ASGI stack -- routing, pydantic validation and response
serialisation -- and only replace the model, the database write and the drift report.
"""

import io
from typing import Any

from fastapi import BackgroundTasks
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
    # The endpoint must hand the report a task collector rather than running it
    # inline: an inline report would put two table reads and the Evidently run into
    # the request path, and could fail an otherwise successful prediction.
    assert isinstance(stubs.report_calls[0]["background_tasks"], BackgroundTasks)


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


def test_latest_drift_report_is_served_and_404s_when_absent(
    client: Any, reports_dir: Any
) -> None:
    """Verify the stored Evidently HTML report is served, and missing means 404."""
    assert client.get("/drift_report/latest?queue=labeled_queue").status_code == 404

    (reports_dir / "latest_labeled_queue.html").write_text("<html>report</html>")
    (reports_dir / "drift_labeled_queue_20260201T120000Z.html").write_text("<html/>")

    response = client.get("/drift_report/latest?queue=labeled_queue")

    assert response.status_code == 200
    assert response.text == "<html>report</html>"
    assert response.headers["content-type"].startswith("text/html")

    history = client.get("/drift_report/history?queue=labeled_queue").json()
    assert [entry["filename"] for entry in history] == [
        "drift_labeled_queue_20260201T120000Z.html"
    ]


def test_manual_trigger_schedules_once(
    client: Any, app_module: Any, monkeypatch: Any, reports_dir: Any
) -> None:
    """Verify the on-demand endpoint schedules a report and refuses to overlap."""
    scheduled: list[dict[str, Any]] = []
    monkeypatch.setattr(
        app_module, "run_drift_report", lambda **kwargs: scheduled.append(kwargs)
    )

    response = client.post("/drift_report/trigger?queue=not_labeled_queue")

    assert response.status_code == 202
    assert response.json() == {"status": "scheduled", "queue": "not_labeled_queue"}
    # TestClient runs background tasks inside the request, so the task already ran.
    assert scheduled[0]["counter_name"] == "not_labeled_queue"
    # A debug run must not be able to start a retraining deployment.
    assert scheduled[0]["trigger_retraining"] is False

    # The endpoint claims the slot and run_drift_report releases it in its finally.
    # The stub above replaced that function, so the slot is still held here -- which
    # is exactly the state a second request must be refused in.
    from drift_report import release_report_slot

    try:
        assert client.post("/drift_report/trigger").status_code == 409
        assert len(scheduled) == 1
    finally:
        release_report_slot()
