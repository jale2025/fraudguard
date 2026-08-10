"""Unit tests for the request models in ``webservice.data_model``."""

import pytest
from data_model import TransactionKnownLabel, TransactionUnknownLabel
from pydantic import ValidationError


def test_known_label_accepts_class_alias(transaction: dict[str, float]) -> None:
    """Verify the reserved word ``class`` is accepted on input and kept out of dumps."""
    transaction["class"] = 1

    parsed = TransactionKnownLabel(**transaction)

    # Clients send "class"; the attribute is target_class because "class" is a
    # reserved keyword. api_to_database.forward_to_database relies on model_dump()
    # returning the field name so it can rename it back for the database.
    assert parsed.target_class == 1
    assert "target_class" in parsed.model_dump()
    assert "class" not in parsed.model_dump()
    assert "class" in parsed.model_dump(by_alias=True)


def test_incomplete_transaction_is_rejected(transaction: dict[str, float]) -> None:
    """Verify that a missing feature is reported instead of reaching the model."""
    del transaction["pc_7"]

    with pytest.raises(ValidationError) as error:
        TransactionUnknownLabel(**transaction)

    assert [detail["loc"][-1] for detail in error.value.errors()] == ["pc_7"]
