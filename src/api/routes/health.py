"""
Health Check Routes

Provides health, readiness, and liveness endpoints for the API.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, status
from loguru import logger

from src.api.schemas import HealthStatus, ReadinessStatus

router = APIRouter(prefix="/health", tags=["Health"])

# Version info
API_VERSION = "1.0.0"

# Global state for model status (will be set by app startup)
_model_status: dict[str, bool] = {
    "fraud_detector": False,
    "price_predictor": False,
    "churn_predictor": False,
}


def set_model_status(model_name: str, loaded: bool) -> None:
    """Update model loading status."""
    _model_status[model_name] = loaded
    logger.info(f"Model status updated: {model_name} = {loaded}")


def get_model_status() -> dict[str, bool]:
    """Get current model loading status."""
    return _model_status.copy()


@router.get(
    "",
    response_model=HealthStatus,
    summary="Health Check",
    description="Returns the health status of the service.",
)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.

    Returns service status, version, and model loading status.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        version=API_VERSION,
        models_loaded=get_model_status(),
    )


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Kubernetes liveness probe endpoint.",
)
async def liveness() -> dict[str, str]:
    """
    Liveness probe for Kubernetes.

    Returns 200 if the service is running.
    """
    return {"status": "alive"}


@router.get(
    "/ready",
    response_model=ReadinessStatus,
    summary="Readiness Probe",
    description="Kubernetes readiness probe endpoint.",
)
async def readiness() -> ReadinessStatus:
    """
    Readiness probe for Kubernetes.

    Checks if all required components are ready to serve traffic.
    """
    checks: dict[str, bool] = {}

    # Check if at least one model is loaded
    model_status = get_model_status()
    checks["models_available"] = any(model_status.values())

    # Check individual models
    for model_name, loaded in model_status.items():
        checks[f"model_{model_name}"] = loaded

    # Overall readiness
    is_ready = checks.get("models_available", False)

    return ReadinessStatus(
        ready=is_ready,
        checks=checks,
    )


@router.get(
    "/version",
    summary="Version Info",
    description="Returns API version information.",
)
async def version_info() -> dict[str, Any]:
    """
    Get API version and build information.
    """
    return {
        "api_version": API_VERSION,
        "service": "ml-observability-platform",
        "timestamp": datetime.now().isoformat(),
    }
