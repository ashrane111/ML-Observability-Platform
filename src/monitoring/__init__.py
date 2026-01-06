"""
Monitoring Module

Provides ML model monitoring capabilities:
- Drift detection using Evidently AI
- Prometheus metrics collection
- Alert management
"""

from src.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertType,
    get_alert_manager,
)
from src.monitoring.drift_detector import (
    DataQualityResult,
    DriftDetector,
    DriftResult,
    DriftStatus,
    create_drift_detector,
)
from src.monitoring.metrics import MetricsCollector, create_metrics_collector, get_metrics

__all__ = [
    # Drift Detection
    "DriftDetector",
    "DriftResult",
    "DriftStatus",
    "DataQualityResult",
    "create_drift_detector",
    # Metrics
    "MetricsCollector",
    "create_metrics_collector",
    "get_metrics",
    # Alerts
    "Alert",
    "AlertManager",
    "AlertRule",
    "AlertType",
    "AlertSeverity",
    "AlertStatus",
    "get_alert_manager",
]
