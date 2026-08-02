"""
Prometheus metrics for observability.

Tracks:
  - Request rate, error rate
  - Inference latency (p50/p95/p99)
  - Kafka consumer lag
  - Feature store latency
  - Model prediction distribution
  - APPROVE/REVIEW/BLOCK ratios
  - Fraud alert rate
  - Drift metrics, model version
"""

from __future__ import annotations

import logging

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Metric Definitions
# ──────────────────────────────────────────────────────

API_UP = Gauge(
    "fraud_api_up",
    "API readiness status (1=ready, 0=not ready)",
)

# Request metrics
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)

ERROR_COUNT = Counter(
    "fraud_api_errors_total",
    "Total API errors",
    ["method", "endpoint", "error_type"],
)

# Inference metrics
INFERENCE_LATENCY = Histogram(
    "fraud_inference_latency_seconds",
    "Model inference latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
)

FEATURE_LATENCY = Histogram(
    "fraud_feature_latency_seconds",
    "Feature generation latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05],
)

PREDICTION_DISTRIBUTION = Histogram(
    "fraud_prediction_probability",
    "Distribution of fraud probabilities",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
)

RISK_SCORE_DISTRIBUTION = Histogram(
    "fraud_risk_score",
    "Distribution of risk scores",
    buckets=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
)

# Decision metrics
DECISION_COUNT = Counter(
    "fraud_decisions_total",
    "Total decisions by type",
    ["decision"],
)

FRAUD_ALERTS = Counter(
    "fraud_alerts_total",
    "Total fraud alerts generated",
)

# Model metrics
MODEL_VERSION = Info(
    "fraud_model",
    "Current model information",
)

KAFKA_CONSUMER_LAG = Gauge(
    "fraud_kafka_consumer_lag",
    "Kafka consumer lag",
    ["topic", "partition"],
)

FEATURE_STORE_LATENCY = Histogram(
    "fraud_feature_store_latency_seconds",
    "Feature store retrieval latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05],
)

DRIFT_SCORE = Gauge(
    "fraud_drift_score",
    "Current drift score",
    ["feature", "metric"],
)

TRANSACTIONS_PROCESSED = Counter(
    "fraud_transactions_processed_total",
    "Total transactions processed",
)


# ──────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────


def record_prediction(
    probability: float,
    risk_score: int,
    decision: str,
    inference_latency_s: float,
) -> None:
    """Record metrics for a single prediction."""
    PREDICTION_DISTRIBUTION.observe(probability)
    RISK_SCORE_DISTRIBUTION.observe(risk_score)
    INFERENCE_LATENCY.observe(inference_latency_s)
    DECISION_COUNT.labels(decision=decision).inc()
    TRANSACTIONS_PROCESSED.inc()

    if decision == "BLOCK":
        FRAUD_ALERTS.inc()


def record_request(method: str, endpoint: str, status: int, latency_s: float) -> None:
    """Record API request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency_s)


def record_error(method: str, endpoint: str, error_type: str) -> None:
    """Record an API error."""
    ERROR_COUNT.labels(method=method, endpoint=endpoint, error_type=error_type).inc()


def set_model_info(version: str, features: int, calibration: str) -> None:
    """Update model info gauge."""
    MODEL_VERSION.info(
        {
            "version": version,
            "feature_count": str(features),
            "calibration": calibration,
        }
    )


def set_api_up(ready: bool = True) -> None:
    """Update API readiness status metric."""
    API_UP.set(1.0 if ready else 0.0)


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest()
