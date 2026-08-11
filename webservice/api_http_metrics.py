from prometheus_client import Counter

PREDICTION_REQUESTS = Counter(
    "fraudguard_prediction_requests_total",
    "Number of requests to FraudGuard prediction endpoints.",
    ["endpoint", "result"],
)

# Endpoints that accept a single JSON transaction. They cannot report invalid
# input: a malformed body is rejected by FastAPI's request validation before the
# handler body runs.
SINGLE_ENDPOINTS = (
    "predict_single_unknown_label",
    "predict_single_known_label",
)

# Endpoints that accept a Parquet upload and validate it themselves.
FILE_ENDPOINTS = (
    "predict_file_unknown_label",
    "predict_file_known_label",
)

RESULTS = ("success", "model_unavailable", "internal_error")
FILE_ONLY_RESULTS = ("invalid_input",)


def initialise_children() -> None:
    """
    Export every reachable (endpoint, result) child at zero from process start.

    A labelled Counter child does not exist in the registry until ``.labels()`` is
    called for it, so it is first scraped already holding 1. Range queries such as
    ``increase()`` and ``rate()`` measure the delta between the first and last
    sample inside their window, which means that very first request has no earlier
    sample to be measured against and is silently lost: an endpoint that was hit
    once stays at 0 on the dashboards forever, and an endpoint hit twice reports 1.
    Creating the children up front gives those queries the zero baseline they need.
    """
    for endpoint in SINGLE_ENDPOINTS + FILE_ENDPOINTS:
        results = RESULTS
        if endpoint in FILE_ENDPOINTS:
            results = results + FILE_ONLY_RESULTS

        for result in results:
            PREDICTION_REQUESTS.labels(endpoint=endpoint, result=result)


initialise_children()
