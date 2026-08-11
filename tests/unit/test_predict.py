"""Unit tests for the inference helper in ``webservice.predict``."""

from typing import Any

import numpy as np
import pandas as pd
import predict as predict_module
import pytest
from data_model import TransactionKnownLabel
from predict import predict


class FakeModel:
    """Stand-in for the MLflow model that records what it was asked to score."""

    def __init__(self) -> None:
        """Prepare the recorder."""
        self.scored: Any = None

    def predict(self, frame: Any) -> np.ndarray:
        """Record the frame and return a fixed fraud prediction."""
        self.scored = frame
        return np.array([1])


def test_predict_drops_label_and_returns_int(
    transaction: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the ground truth never reaches the model and a plain int comes back."""
    transaction["class"] = 1
    model = FakeModel()

    # Faking at this seam keeps the lru_cache on load_model out of the picture.
    monkeypatch.setattr(predict_module, "ensure_model_available", lambda *args: model)

    result = predict("test_model", TransactionKnownLabel(**transaction), "production")

    assert result == 1
    assert isinstance(result, int)
    assert "target_class" not in model.scored.columns
    assert "class" not in model.scored.columns
    assert len(model.scored.columns) == 30
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in model.scored.dtypes)
