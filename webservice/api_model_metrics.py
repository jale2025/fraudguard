from collections import Counter as FrequencyCounter

from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    "fraudguard_predictions_total",
    "Number of transactions scored by the model.",
    ["request_type", "ground_truth", "prediction"],
)

CLASSIFICATION_OUTCOMES = Counter(
    "fraudguard_classification_outcomes_total",
    "Classification outcomes for transactions with known labels.",
    ["request_type", "outcome"],
)

FILE_ROWS = Histogram(
    "fraudguard_prediction_file_rows",
    "Number of transactions processed per uploaded file.",
    ["ground_truth"],
    # buckets=(1, 10, 50, 100, 250, 500),
)

MODEL_INFERENCE_DURATION = Histogram(
    "fraudguard_model_inference_duration_seconds",
    "Time spent performing model inference.",
    ["request_type"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)

PREDICTION_NAMES = {
    0: "legitimate",
    1: "fraud",
}

OUTCOME_NAMES = {
    (0, 0): "true_negative",
    (0, 1): "false_positive",
    (1, 0): "false_negative",
    (1, 1): "true_positive",
}


def record_prediction_metrics(
    predictions: int | list[int],
    *,
    request_type: str,
    ground_truth: str,
    actual: int | list[int] | None = None,
) -> None:
    """Record model outputs and, if available, classification outcomes."""
    prediction_values = (
        [int(predictions)]
        if isinstance(predictions, int)
        else [int(value) for value in predictions]
    )

    prediction_counts = FrequencyCounter(prediction_values)

    for prediction, count in prediction_counts.items():
        prediction_name = PREDICTION_NAMES.get(prediction, "other")

        PREDICTIONS.labels(
            request_type=request_type,
            ground_truth=ground_truth,
            prediction=prediction_name,
        ).inc(count)

    if request_type == "file":
        FILE_ROWS.labels(
            ground_truth=ground_truth,
        ).observe(len(prediction_values))

    if actual is None:
        return

    actual_values = (
        [int(actual)] if isinstance(actual, int) else [int(value) for value in actual]
    )

    if len(actual_values) != len(prediction_values):
        raise ValueError("Actual labels and predictions have different lengths.")

    outcome_counts = FrequencyCounter(
        zip(actual_values, prediction_values, strict=True)
    )

    for outcome, count in outcome_counts.items():
        outcome_name = OUTCOME_NAMES.get(outcome, "other")

        CLASSIFICATION_OUTCOMES.labels(
            request_type=request_type,
            outcome=outcome_name,
        ).inc(count)
