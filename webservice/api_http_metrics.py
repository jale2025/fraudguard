from prometheus_client import Counter, Gauge

PREDICTION_REQUESTS = Counter(
    "fraudguard_prediction_requests_total",
    "Number of requests to FraudGuard prediction endpoints.",
    ["endpoint", "result"],
)