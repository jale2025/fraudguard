"""Unit tests for the file helpers in ``src.data_helper``."""

from pathlib import Path

import pytest

from src.data_helper import is_parquet, move_file, validate_data_files


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("transactions.parquet", True),
        ("transactions.PARQUET", True),
        ("transactions.csv", False),
        ("transactions", False),
    ],
)
def test_is_parquet(file_name: str, expected: bool) -> None:
    """Verify that the parquet check accepts any case of the suffix, and nothing else."""
    assert is_parquet(file_name) is expected


def test_validate_data_files_splits_valid_and_invalid(tmp_path: Path) -> None:
    """Verify that parquet files are accepted for ingestion and other files are not."""
    valid = tmp_path / "transactions.parquet"
    invalid = tmp_path / "notes.txt"
    valid.touch()
    invalid.touch()

    result = validate_data_files(tmp_path)

    assert result.valid_list == [str(valid)]
    assert result.invalid_list == [str(invalid)]


def test_move_file_keeps_content_and_does_not_overwrite(tmp_path: Path) -> None:
    """Verify that moving two identically named files keeps both, with their content."""
    destination = tmp_path / "processed"

    for content in (b"first", b"second"):
        source = tmp_path / "transactions.parquet"
        source.write_bytes(content)
        move_file(source, destination)

    moved = sorted(destination.iterdir())

    assert not (tmp_path / "transactions.parquet").exists()
    assert len(moved) == 2
    assert {path.read_bytes() for path in moved} == {b"first", b"second"}
    assert all(path.stem.startswith("transactions_") for path in moved)
    assert all(path.suffix == ".parquet" for path in moved)
