"""
Prometheus Metrics Module

Exposes application metrics for Prometheus scraping.
"""

import time

from fastapi import APIRouter, Response

# from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

# from functools import wraps
# from typing import Callable


router = APIRouter(tags=["Metrics"])

# ============================================================================
# Metric Definitions
# ============================================================================

# Application Info
APP_INFO = Info("ml_observability_app", "ML Observability Platform information")
APP_INFO.info(
    {
        "version": "1.0.0",
        "service": "ml-observability-platform",
    }
)

# Request Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Prediction Metrics
PREDICTION_COUNT = Counter("ml_predictions_total", "Total ML predictions", ["model_type", "status"])

PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds",
    "ML prediction latency in seconds",
    ["model_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_PROBABILITY = Histogram(
    "ml_prediction_probability",
    "Distribution of prediction probabilities",
    ["model_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Model Metrics
MODEL_LOADED = Gauge("ml_model_loaded", "Whether model is loaded (1) or not (0)", ["model_type"])

# Drift Metrics
DRIFT_DETECTED = Gauge(
    "ml_drift_detected", "Whether drift is detected (1) or not (0)", ["model_type"]
)

DRIFT_SCORE = Gauge("ml_drift_score", "Current drift score (PSI)", ["model_type"])

# Error Metrics
ERROR_COUNT = Counter("ml_errors_total", "Total errors", ["error_type", "model_type"])


# ============================================================================
# Metric Recording Functions
# ============================================================================


def record_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record HTTP request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_prediction(model_type: str, success: bool, latency_ms: float, probability: float = None):
    """Record prediction metrics."""
    status = "success" if success else "error"
    PREDICTION_COUNT.labels(model_type=model_type, status=status).inc()
    PREDICTION_LATENCY.labels(model_type=model_type).observe(
        latency_ms / 1000
    )  # Convert to seconds

    if probability is not None and success:
        PREDICTION_PROBABILITY.labels(model_type=model_type).observe(probability)


def set_model_loaded(model_type: str, loaded: bool):
    """Set model loaded status."""
    MODEL_LOADED.labels(model_type=model_type).set(1 if loaded else 0)


def set_drift_status(model_type: str, detected: bool, score: float = 0.0):
    """Set drift detection status."""
    DRIFT_DETECTED.labels(model_type=model_type).set(1 if detected else 0)
    DRIFT_SCORE.labels(model_type=model_type).set(score)


def record_error(error_type: str, model_type: str = "unknown"):
    """Record an error."""
    ERROR_COUNT.labels(error_type=error_type, model_type=model_type).inc()


# ============================================================================
# Metrics Endpoint
# ============================================================================


@router.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Middleware Helper
# ============================================================================


class MetricsMiddleware:
    """Middleware to automatically record request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()

        # Track response status
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "/")

            # Skip metrics endpoint itself to avoid recursion
            if path != "/metrics":
                record_request(method, path, status_code, duration)
