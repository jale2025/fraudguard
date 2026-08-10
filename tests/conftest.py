"""
Shared pytest configuration and fixtures.

Several modules read ``os.environ[...]`` at *import* time and freeze the values into
their default arguments (``src/train.py``, ``src/model_registration.py``,
``prefect/tasks.py``, ``webservice/app.py``). The environment therefore has to exist
before any test module is imported, which is why this runs in the module body instead
of an autouse fixture.

``webservice/app.py`` calls ``load_dotenv()`` with the default ``override=False``, so
the values below also win over a developer's local ``.env``. Every host name uses the
reserved ``.invalid`` TLD: a test that forgets to replace a collaborator then fails on
name resolution instead of quietly talking to the running dev stack.
"""

import os

import numpy as np
import pandas as pd
import pytest

TEST_ENV = {
    "DB_URI": "postgresql://test_user:test_pw@db.invalid:5432/test_db",
    "DBT_SCHEMA": "dev",
    "MLFLOW_TRACKING_URI": "http://mlflow.invalid:5000",
    "MODEL_NAME": "test_model",
    "DEFAULT_MODEL_ALIAS": "test_production",
    "REDIS_HOST": "redis.invalid",
    "DATA_DIR_INCOMING": "/tmp/fraudguard/incoming",
    "DATA_DIR_PROCESSED": "/tmp/fraudguard/processed",
    "DATA_DIR_QUARANTINE": "/tmp/fraudguard/quarantine",
    "DBT_PROJECT_DIR": "/tmp/fraudguard/dbt",
}

for _key, _value in TEST_ENV.items():
    os.environ[_key] = _value


@pytest.fixture
def transaction() -> dict[str, float]:
    """Return a complete request body holding all 30 transaction features."""
    body: dict[str, float] = {"elapsed_sec": 2.0, "amount": 255.65}
    body.update({f"pc_{index}": 0.1 * index for index in range(1, 29)})
    return body


@pytest.fixture
def raw_transactions():
    """Return a factory for frames using the raw source column names."""

    def _make(rows: int = 3, with_label: bool = False) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "Time": np.arange(rows, dtype=float),
                **{f"V{index}": np.full(rows, 0.1 * index) for index in range(1, 29)},
                "Amount": np.linspace(10.0, 100.0, rows),
            }
        )
        if with_label:
            frame["Class"] = np.resize([0, 1], rows).astype(int)
        return frame

    return _make
