"""
Unit tests for the background drift-report scheduling.

The point of the module under test is that the prediction endpoints never pay for,
and never fail because of, the drift report. These tests therefore assert two things
above all: that nothing propagates out of either entry point, and that the outcome
is visible on the Prometheus counters instead.
"""

import os
from pathlib import Path
from typing import Any

import drift_report
import pytest
from prometheus_client import REGISTRY

RUNS_METRIC = "fraudguard_drift_report_runs_total"


class FakeRedis:
    """Minimal stand-in recording what the scheduler did to the counter."""

    def __init__(self, value: str | None, error: Exception | None = None) -> None:
        """Store the counter value to serve and an optional failure to raise."""
        self.value = value
        self.error = error
        self.sets: list[tuple[str, int]] = []

    def get(self, name: str) -> str | None:
        """Return the stored counter, or raise when the test asked for a failure."""
        if self.error is not None:
            raise self.error
        return self.value

    def set(self, name: str, value: int) -> None:
        """Record a counter reset."""
        self.sets.append((name, value))


class FakeBackgroundTasks:
    """Stand-in for ``fastapi.BackgroundTasks`` that only records scheduled work."""

    def __init__(self) -> None:
        """Start with no scheduled work."""
        self.tasks: list[dict[str, Any]] = []

    def add_task(self, func: Any, **kwargs: Any) -> None:
        """Record a scheduled task instead of running it."""
        self.tasks.append({"func": func, **kwargs})


def runs_total(queue: str, result: str) -> float:
    """Return the current value of one drift-report outcome counter child."""
    value = REGISTRY.get_sample_value(RUNS_METRIC, {"queue": queue, "result": result})
    # initialise_children() creates every child, so a None here is a real failure.
    assert value is not None, f"missing counter child {queue}/{result}"
    return value


@pytest.fixture(autouse=True)
def released_slot():
    """Guarantee a free report slot before and after every test."""
    drift_report.release_report_slot()
    yield
    drift_report.release_report_slot()


@pytest.fixture
def redis_stub(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that installs a FakeRedis as the module's client."""

    def _install(value: str | None, error: Exception | None = None) -> FakeRedis:
        fake = FakeRedis(value=value, error=error)
        monkeypatch.setattr(drift_report, "redis_client", lambda: fake)
        return fake

    return _install


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a temporary reports directory."""
    monkeypatch.setattr(drift_report, "REPORTS_DIR", tmp_path)
    return tmp_path


def schedule(counter_name: str = "labeled_queue") -> FakeBackgroundTasks:
    """Run the scheduler against a fresh background-task collector."""
    tasks = FakeBackgroundTasks()
    drift_report.create_report_and_trigger_workflow(
        counter_name=counter_name,
        current_data_query=drift_report.QUEUE_QUERIES[counter_name],
        background_tasks=tasks,
    )
    return tasks


def test_missing_counter_key_schedules_nothing(redis_stub: Any) -> None:
    """Verify a redis key that was never created is treated as zero, not a crash."""
    # Nothing seeds these keys -- only INCR creates them -- so a cold redis used to
    # raise TypeError from int(None) and turn a good prediction into a 500.
    fake = redis_stub(None)

    tasks = schedule()

    assert tasks.tasks == []
    assert fake.sets == []


def test_counter_below_threshold_schedules_nothing(redis_stub: Any) -> None:
    """Verify the report waits until the threshold is actually reached."""
    fake = redis_stub(str(drift_report.THRESHOLD_QUEUE_COUNTER - 1))

    tasks = schedule()

    assert tasks.tasks == []
    assert fake.sets == []


def test_counter_at_threshold_schedules_and_resets(redis_stub: Any) -> None:
    """Verify the report is scheduled as a background task and the counter reset."""
    fake = redis_stub(str(drift_report.THRESHOLD_QUEUE_COUNTER))

    tasks = schedule("not_labeled_queue")

    assert len(tasks.tasks) == 1
    scheduled = tasks.tasks[0]
    assert scheduled["func"] is drift_report.run_drift_report
    assert scheduled["counter_name"] == "not_labeled_queue"
    assert (
        scheduled["current_data_query"]
        == (drift_report.QUEUE_QUERIES["not_labeled_queue"])
    )
    # Reset happens immediately, so requests arriving during the run are not all
    # counted as skipped.
    assert fake.sets == [("not_labeled_queue", 0)]


def test_second_schedule_is_skipped_while_a_report_runs(redis_stub: Any) -> None:
    """Verify the busy guard stops a burst from queueing one report per request."""
    redis_stub(str(drift_report.THRESHOLD_QUEUE_COUNTER + 100))
    before = runs_total("labeled_queue", "skipped_busy")

    first = schedule()
    second = schedule()

    assert len(first.tasks) == 1
    assert second.tasks == []
    assert runs_total("labeled_queue", "skipped_busy") == before + 1


def test_unreadable_counter_does_not_raise(redis_stub: Any) -> None:
    """Verify an unreachable redis is recorded, not propagated to the caller."""
    redis_stub(None, error=ConnectionError("redis is down"))
    before = runs_total("labeled_queue", "counter_unavailable")

    tasks = schedule()

    assert tasks.tasks == []
    assert runs_total("labeled_queue", "counter_unavailable") == before + 1
    # The slot must be free again, otherwise one redis blip disables the report.
    assert drift_report.claim_report_slot() is True


def test_failing_report_is_contained(
    monkeypatch: pytest.MonkeyPatch, reports_dir: Path
) -> None:
    """Verify a crashing report is recorded as a failure and releases the slot."""

    def boom(**kwargs: Any) -> None:
        raise RuntimeError("database is gone")

    monkeypatch.setattr(drift_report, "get_data_from_postgresql_db", boom)
    before = runs_total("labeled_queue", "failure")
    assert drift_report.claim_report_slot() is True

    drift_report.run_drift_report(
        counter_name="labeled_queue",
        current_data_query="SELECT 1",
    )

    assert runs_total("labeled_queue", "failure") == before + 1
    assert REGISTRY.get_sample_value("fraudguard_drift_report_in_progress", {}) == 0.0
    assert drift_report.claim_report_slot() is True


def test_successful_report_writes_html_and_sets_gauges(
    monkeypatch: pytest.MonkeyPatch, reports_dir: Path
) -> None:
    """
    Verify the whole background path runs, HTML included.

    This drives the real Evidently report rather than stubbing it, because the HTML
    file is the deliverable the Grafana dashboard links to -- a renamed Evidently
    helper would otherwise only show up in production.

    Rendering the report takes minutes even for a single feature, which is the whole
    reason this work no longer sits in the request path. Set TEST_EVIDENTLY=1 to run
    it; leaving it unset keeps a plain ``pytest`` fast.
    """
    if not os.environ.get("TEST_EVIDENTLY"):
        pytest.skip("TEST_EVIDENTLY is not set; skipping the real Evidently run.")

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"pc_1": rng.normal(size=200)})
    # A large mean shift, so the run also exercises the drift-detected branch.
    current = pd.DataFrame({"pc_1": rng.normal(loc=6.0, size=200), "prediction": 0})

    frames = {
        "current": current,
        drift_report.REFERENCE_DATA_QUERY: reference,
    }
    monkeypatch.setattr(
        drift_report,
        "get_data_from_postgresql_db",
        lambda db_uri, query: frames.get(query, frames["current"]).copy(),
    )
    triggered: list[dict[str, Any]] = []
    monkeypatch.setattr(
        drift_report,
        "_trigger_retraining_deployment",
        lambda: triggered.append({}),
    )
    before = runs_total("labeled_queue", "success")

    drift_report.run_drift_report(
        counter_name="labeled_queue", current_data_query="current"
    )

    assert runs_total("labeled_queue", "success") == before + 1
    assert triggered == [{}]
    assert REGISTRY.get_sample_value("fraudguard_data_drift_detected", {}) == 1.0
    assert (
        REGISTRY.get_sample_value(
            "fraudguard_feature_drift_score", {"feature_name": "pc_1"}
        )
        is not None
    )

    latest = reports_dir / "latest_labeled_queue.html"
    assert latest.is_file()
    assert latest.stat().st_size > 0
    assert "<html" in latest.read_text(encoding="utf-8")[:2000].lower()
    # The timestamped copy is kept alongside the stable "latest" one. That the run
    # succeeded at all also proves the serving-only "prediction" column was dropped:
    # it is absent from the reference frame and would otherwise be a schema mismatch.
    assert len(list(reports_dir.glob("drift_labeled_queue_*.html"))) == 1


def test_prune_keeps_only_the_newest_reports(reports_dir: Path) -> None:
    """Verify retention keeps the configured number of timestamped reports."""
    stamps = [f"2026081{index}T120000Z" for index in range(8)]
    for stamp in stamps:
        (reports_dir / f"drift_labeled_queue_{stamp}.html").write_text("x")
    # A report for the other queue must survive untouched.
    (reports_dir / "drift_not_labeled_queue_20260101T120000Z.html").write_text("x")

    drift_report._prune_reports("labeled_queue")

    remaining = sorted(path.name for path in reports_dir.glob("drift_labeled_queue_*"))
    assert len(remaining) == drift_report.MAX_REPORTS_PER_QUEUE
    assert remaining == [
        f"drift_labeled_queue_{stamp}.html"
        for stamp in stamps[-drift_report.MAX_REPORTS_PER_QUEUE :]
    ]
    assert (reports_dir / "drift_not_labeled_queue_20260101T120000Z.html").exists()


def test_list_reports_is_newest_first(reports_dir: Path) -> None:
    """Verify the history listing is ordered and labelled per queue."""
    (reports_dir / "drift_labeled_queue_20260101T120000Z.html").write_text("old")
    (reports_dir / "drift_labeled_queue_20260201T120000Z.html").write_text("new")

    listed = drift_report.list_reports("labeled_queue")

    assert [entry["created_at"] for entry in listed] == [
        "20260201T120000Z",
        "20260101T120000Z",
    ]
    assert {entry["queue"] for entry in listed} == {"labeled_queue"}


def test_report_file_rejects_traversal(reports_dir: Path) -> None:
    """Verify a report name cannot be used to read files outside the reports dir."""
    outside = reports_dir.parent / "secret.html"
    outside.write_text("secret")

    assert drift_report.report_file("../secret.html") is None
    assert drift_report.report_file("does_not_exist.html") is None

    (reports_dir / "latest_labeled_queue.html").write_text("report")
    assert drift_report.latest_report_file("labeled_queue") is not None
