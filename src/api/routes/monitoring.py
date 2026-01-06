"""
Monitoring Routes

Provides endpoints for drift detection, data quality, and alerts.
"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from loguru import logger

from src.api.schemas import (
    AlertAcknowledgeRequest,
    AlertListResponse,
    AlertResolveRequest,
    AlertResponse,
    AlertSeverityEnum,
    AlertSummaryResponse,
    DataQualityRequest,
    DataQualityResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    DriftStatusEnum,
    FeatureDriftInfo,
    ModelType,
)
from src.monitoring.alerts import AlertSeverity, get_alert_manager
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics import get_metrics

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Global drift detectors (will be set by app startup)
_drift_detectors: dict[str, Optional[DriftDetector]] = {
    "fraud": None,
    "price": None,
    "churn": None,
}


def set_drift_detector(model_type: str, detector: DriftDetector) -> None:
    """Set a drift detector for a model type."""
    _drift_detectors[model_type] = detector
    logger.info(f"Drift detector set for {model_type}")


def get_drift_detector(model_type: str) -> Optional[DriftDetector]:
    """Get a drift detector by model type."""
    return _drift_detectors.get(model_type)


# =============================================================================
# Drift Detection Endpoints
# =============================================================================


@router.post(
    "/drift/check",
    response_model=DriftCheckResponse,
    summary="Check Data Drift",
    description="Check for data drift in the provided dataset.",
)
async def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """
    Check for data drift between reference and current data.

    Compares the provided data against the reference dataset
    and returns drift metrics for each feature.
    """
    model_type = request.model_type.value
    detector = get_drift_detector(model_type)

    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Drift detector not initialized for {model_type}",
        )

    try:
        # Convert to DataFrame
        current_data = pd.DataFrame(request.data)

        # Run drift detection
        result = detector.detect_drift(
            current_data,
            dataset_name=request.dataset_name or "current",
        )

        # Build feature drift info
        feature_drift = [
            FeatureDriftInfo(
                feature_name=feature,
                drift_score=score,
                is_drifted=feature in result.drifted_features,
            )
            for feature, score in result.feature_drift_scores.items()
        ]

        # Create alert if drift detected
        if result.drift_status.value in ["warning", "critical"]:
            alert_manager = get_alert_manager()
            alert_manager.create_drift_alert(
                model_name=model_type,
                drift_share=result.drift_share,
                drifted_features=result.drifted_features,
                dataset_name=request.dataset_name or "current",
                feature_scores=result.feature_drift_scores,
            )

        return DriftCheckResponse(
            drift_status=DriftStatusEnum(result.drift_status.value),
            dataset_drift_detected=result.dataset_drift_detected,
            drift_share=result.drift_share,
            drifted_features_count=result.number_of_drifted_features,
            total_features=result.total_features,
            feature_drift=feature_drift,
            timestamp=result.timestamp,
            model_type=request.model_type,
        )

    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift check failed: {str(e)}",
        )


@router.get(
    "/drift/status/{model_type}",
    summary="Get Drift Status",
    description="Get the current drift status for a model.",
)
async def get_drift_status(model_type: ModelType) -> dict[str, Any]:
    """
    Get the current drift detection status for a model.

    Returns the last known drift metrics.
    """
    detector = get_drift_detector(model_type.value)

    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Drift detector not initialized for {model_type.value}",
        )

    return {
        "model_type": model_type.value,
        "detector_configured": True,
        "psi_threshold_warning": detector.psi_threshold_warning,
        "psi_threshold_critical": detector.psi_threshold_critical,
        "drift_share_threshold": detector.drift_share_threshold,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# Data Quality Endpoints
# =============================================================================


@router.post(
    "/quality/check",
    response_model=DataQualityResponse,
    summary="Check Data Quality",
    description="Check data quality metrics for the provided dataset.",
)
async def check_data_quality(request: DataQualityRequest) -> DataQualityResponse:
    """
    Check data quality metrics.

    Returns information about missing values, duplicates, and other quality issues.
    """
    model_type = request.model_type.value
    detector = get_drift_detector(model_type)

    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Drift detector not initialized for {model_type}",
        )

    try:
        # Convert to DataFrame
        data = pd.DataFrame(request.data)

        # Run quality check
        result = detector.check_data_quality(
            data,
            dataset_name=request.dataset_name or "current",
        )

        return DataQualityResponse(
            dataset_name=result.dataset_name,
            total_rows=result.total_rows,
            missing_values_share=result.missing_values_share,
            columns_with_missing=result.columns_with_missing,
            duplicate_rows=result.duplicate_rows,
            timestamp=result.timestamp,
        )

    except Exception as e:
        logger.error(f"Data quality check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data quality check failed: {str(e)}",
        )


# =============================================================================
# Alert Endpoints
# =============================================================================


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List Alerts",
    description="Get list of alerts with optional filtering.",
)
async def list_alerts(
    model_name: Optional[str] = None,
    severity: Optional[AlertSeverityEnum] = None,
    active_only: bool = True,
    limit: int = 100,
) -> AlertListResponse:
    """
    List alerts with optional filtering.

    Can filter by model name, severity, and active status.
    """
    alert_manager = get_alert_manager()

    if active_only:
        alerts = alert_manager.get_active_alerts(
            model_name=model_name,
            severity=AlertSeverity(severity.value) if severity else None,
        )
    else:
        alerts = alert_manager.get_alert_history(
            model_name=model_name,
            limit=limit,
        )

    alert_responses = [
        AlertResponse(
            alert_id=alert.alert_id,
            alert_type=alert.alert_type.value,
            severity=AlertSeverityEnum(alert.severity.value),
            status=alert.status.value,
            model_name=alert.model_name,
            title=alert.title,
            message=alert.message,
            created_at=alert.created_at,
            metric_name=alert.metric_name,
            metric_value=alert.metric_value,
        )
        for alert in alerts[:limit]
    ]

    return AlertListResponse(
        alerts=alert_responses,
        total=len(alert_responses),
    )


@router.get(
    "/alerts/summary",
    response_model=AlertSummaryResponse,
    summary="Alert Summary",
    description="Get summary of current alert status.",
)
async def get_alert_summary(
    model_name: Optional[str] = None,
) -> AlertSummaryResponse:
    """
    Get a summary of current alert status.

    Returns counts by severity and type.
    """
    alert_manager = get_alert_manager()
    summary = alert_manager.get_alert_summary(model_name=model_name)

    return AlertSummaryResponse(
        total_active=summary["total_active"],
        by_severity=summary["by_severity"],
        by_type=summary["by_type"],
        oldest_alert=(
            datetime.fromisoformat(summary["oldest_alert"]) if summary["oldest_alert"] else None
        ),
        newest_alert=(
            datetime.fromisoformat(summary["newest_alert"]) if summary["newest_alert"] else None
        ),
    )


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertResponse,
    summary="Get Alert",
    description="Get details of a specific alert.",
)
async def get_alert(alert_id: str) -> AlertResponse:
    """
    Get details of a specific alert by ID.
    """
    alert_manager = get_alert_manager()
    alert = alert_manager.get_alert_by_id(alert_id)

    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return AlertResponse(
        alert_id=alert.alert_id,
        alert_type=alert.alert_type.value,
        severity=AlertSeverityEnum(alert.severity.value),
        status=alert.status.value,
        model_name=alert.model_name,
        title=alert.title,
        message=alert.message,
        created_at=alert.created_at,
        metric_name=alert.metric_name,
        metric_value=alert.metric_value,
    )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="Acknowledge Alert",
    description="Acknowledge an alert.",
)
async def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest,
) -> AlertResponse:
    """
    Acknowledge an alert.

    Marks the alert as acknowledged but not resolved.
    """
    alert_manager = get_alert_manager()
    alert = alert_manager.acknowledge_alert(
        alert_id,
        acknowledged_by=request.acknowledged_by,
    )

    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return AlertResponse(
        alert_id=alert.alert_id,
        alert_type=alert.alert_type.value,
        severity=AlertSeverityEnum(alert.severity.value),
        status=alert.status.value,
        model_name=alert.model_name,
        title=alert.title,
        message=alert.message,
        created_at=alert.created_at,
        metric_name=alert.metric_name,
        metric_value=alert.metric_value,
    )


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertResponse,
    summary="Resolve Alert",
    description="Resolve an alert.",
)
async def resolve_alert(
    alert_id: str,
    request: AlertResolveRequest,
) -> AlertResponse:
    """
    Resolve an alert.

    Marks the alert as resolved with optional notes.
    """
    alert_manager = get_alert_manager()
    alert = alert_manager.resolve_alert(
        alert_id,
        resolved_by=request.resolved_by,
        resolution_notes=request.resolution_notes,
    )

    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found",
        )

    return AlertResponse(
        alert_id=alert.alert_id,
        alert_type=alert.alert_type.value,
        severity=AlertSeverityEnum(alert.severity.value),
        status=alert.status.value,
        model_name=alert.model_name,
        title=alert.title,
        message=alert.message,
        created_at=alert.created_at,
        metric_name=alert.metric_name,
        metric_value=alert.metric_value,
    )


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================


@router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Get Prometheus metrics.",
)
async def prometheus_metrics() -> Response:
    """
    Expose Prometheus metrics.

    Returns metrics in Prometheus text format.
    """
    metrics_data = get_metrics()
    return Response(
        content=metrics_data,
        media_type="text/plain; charset=utf-8",
    )
