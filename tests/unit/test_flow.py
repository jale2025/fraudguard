"""
Unit tests for the pipeline gating in ``prefect.flow``.

The flow is called through ``.fn()`` so the Prefect engine is bypassed and the plain
function under the decorator runs. Every stage is replaced by a recorder, so what is
under test is the short-circuit chain: a stage may only run once the previous one
reported success.
"""

import logging
from typing import Any

import flow as flow_module
import pytest


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every stage with a recorder and return the shared call log."""
    calls: list[str] = []

    monkeypatch.setattr(
        flow_module, "get_run_logger", lambda: logging.getLogger("test-flow")
    )
    monkeypatch.setattr(flow_module, "data_available", lambda dir_path: True)

    def record(name: str, result: Any):
        def _stage(*args: Any, **kwargs: Any) -> Any:
            label = f"{name}:{kwargs['target']}" if "target" in kwargs else name
            calls.append(label)
            return result

        return _stage

    monkeypatch.setattr(flow_module, "data_ingestion", record("data_ingestion", True))
    monkeypatch.setattr(flow_module, "dbt_build", record("dbt_build", True))
    monkeypatch.setattr(
        flow_module, "get_data_train_model", record("get_data_train_model", {})
    )
    monkeypatch.setattr(
        flow_module, "model_registration", record("model_registration", True)
    )
    return calls


def test_pipeline_runs_all_stages(pipeline: list[str]) -> None:
    """Verify the full chain runs in order and rebuilds prod after a new model."""
    flow_module.fraudguard_pipeline.fn()

    assert pipeline == [
        "data_ingestion",
        "dbt_build:dev",
        "get_data_train_model",
        "model_registration",
        "dbt_build:prod",
    ]


@pytest.mark.xfail(
    raises=UnboundLocalError,
    strict=True,
    reason=(
        "flow.py reads new_model_registered unconditionally, but only assigns it "
        "inside 'if dbt_success:'. Every run without new data therefore crashes."
    ),
)
def test_pipeline_stops_when_no_data_available(
    pipeline: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that an empty incoming folder skips the pipeline instead of failing."""
    monkeypatch.setattr(flow_module, "data_available", lambda dir_path: False)

    flow_module.fraudguard_pipeline.fn()

    assert pipeline == []
