"""
Fixtures for the integration tests.

``import app`` pulls in mlflow, evidently, redis and prefect.deployments and costs
roughly twenty seconds, so it happens lazily inside a session-scoped fixture. Importing
it at module level would make the fast ``tests/unit`` run pay that cost too.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def app_module() -> Any:
    """Import and return the FastAPI application module."""
    import app

    return app


@pytest.fixture(scope="session")
def client(app_module: Any) -> Any:
    """Return a TestClient that serves the app in-process."""
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as test_client:
        yield test_client


@dataclass
class Stubs:
    """Records what the endpoints handed to their out-of-process collaborators."""

    prediction: int = 1
    error: Exception | None = None
    predict_calls: list[dict[str, Any]] = field(default_factory=list)
    forward_calls: list[dict[str, Any]] = field(default_factory=list)
    report_calls: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def stubs(app_module: Any, monkeypatch: pytest.MonkeyPatch) -> Stubs:
    """
    Replace the model, the database write and the drift report with recorders.

    ``app.py`` does ``from predict import predict``, which binds the name into the app
    module's namespace, so ``app.predict`` is the patch target -- patching
    ``predict.predict`` would have no effect.
    """
    recorded = Stubs()

    def fake_predict(model_name: str, data: Any, alias: str) -> Any:
        recorded.predict_calls.append(
            {"model_name": model_name, "data": data, "alias": alias}
        )
        if recorded.error is not None:
            raise recorded.error
        if isinstance(data, pd.DataFrame):
            return [recorded.prediction] * len(data)
        return recorded.prediction

    monkeypatch.setattr(app_module, "predict", fake_predict)
    monkeypatch.setattr(
        app_module,
        "forward_to_database",
        lambda **kwargs: recorded.forward_calls.append(kwargs),
    )
    monkeypatch.setattr(
        app_module,
        "create_report_and_trigger_workflow",
        lambda **kwargs: recorded.report_calls.append(kwargs),
    )
    return recorded


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Point the drift-report module at a temporary reports directory.

    The default is the container's ``/reports`` volume, which does not exist on a
    developer machine, so the report endpoints have to be told where to look.
    """
    import drift_report

    monkeypatch.setattr(drift_report, "REPORTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def postgres_uri() -> str:
    """
    Return a reachable Postgres URI, or skip the test.

    Set TEST_DB_URI to run the database-backed tests, for example after
    ``docker compose up -d db_service``. Leaving it unset keeps a plain local
    ``pytest`` green with nothing running.
    """
    uri = os.environ.get("TEST_DB_URI")
    if not uri:
        pytest.skip("TEST_DB_URI is not set; skipping the database-backed test.")
    return uri
