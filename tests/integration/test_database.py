"""
Integration test for the parquet ingestion against a real Postgres.

Skipped unless TEST_DB_URI points at a database bootstrapped with
``init-db/init-db.sql``.
"""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src import data_helper
from src.data_helper import parquet_to_sql


def test_parquet_to_sql_writes_transactions(
    postgres_uri: str,
    tmp_path: Path,
    raw_transactions: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a parquet file is appended to raw.transactions with its columns intact."""
    file_path = tmp_path / "transactions.parquet"
    raw_transactions(rows=3, with_label=True).to_parquet(file_path, index=False)

    # database_url() reads DB_USER/DB_PASSWORD/... which the global conftest points at
    # an unreachable host on purpose, so redirect it at the database under test.
    monkeypatch.setattr(data_helper, "database_url", lambda: postgres_uri)

    engine = create_engine(postgres_uri)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE raw.transactions"))

    parquet_to_sql(str(file_path), table_name="transactions")

    with engine.connect() as connection:
        rows = connection.execute(
            text('SELECT "Time", "V1", "Amount", "Class" FROM raw.transactions')
        ).all()

    assert len(rows) == 3
    assert [row.Time for row in rows] == [0.0, 1.0, 2.0]
    assert [row.Class for row in rows] == [0, 1, 0]
