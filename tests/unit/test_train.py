"""Unit tests for the training step in ``src.train``."""

import numpy as np
import pandas as pd

from src.train import train_model


def make_training_frame(rows: int = 120) -> pd.DataFrame:
    """Return a small, mildly separable training frame with a balanced target."""
    label = np.resize([0, 1], rows)
    frame = pd.DataFrame(
        {
            "elapsed_sec": np.arange(rows, dtype=float),
            **{f"pc_{index}": np.full(rows, 0.1 * index) for index in range(1, 29)},
            # Fraud gets a clearly higher amount so recall is not degenerate.
            "amount": np.where(label == 1, 900.0, 20.0),
            "class": label,
        }
    )
    return frame


def test_train_model_returns_model_and_metrics() -> None:
    """Verify the training result carries everything the registration step consumes."""
    frame = make_training_frame()

    result = train_model(df_training_data=frame, test_size=0.2)

    assert set(result) == {
        "model",
        "recall_train",
        "recall_test",
        "training_rows",
        "test_rows",
        "input_schema",
        "input_example",
    }
    assert result["training_rows"] + result["test_rows"] == len(frame)
    assert result["test_rows"] == 24
    assert 0.0 <= result["recall_train"] <= 1.0
    assert 0.0 <= result["recall_test"] <= 1.0
    # The registration step calls .best_estimator_ on this object.
    assert hasattr(result["model"], "best_estimator_")
    assert len(result["input_example"]) == 5
