"""Prefect workflows for ML Observability Platform.

Provides automated workflows for:
- Model retraining based on drift detection
- Data quality monitoring
- Scheduled model evaluation
- Alert-triggered remediation
"""

from .retraining import (
    retraining_flow,
    drift_triggered_retraining_flow,
    scheduled_retraining_flow,
)
from .monitoring import (
    data_quality_flow,
    drift_monitoring_flow,
    model_health_flow,
)
from .tasks import (
    load_data,
    train_model,
    evaluate_model,
    check_drift,
    send_alert,
)

__all__ = [
    # Flows
    "retraining_flow",
    "drift_triggered_retraining_flow",
    "scheduled_retraining_flow",
    "data_quality_flow",
    "drift_monitoring_flow",
    "model_health_flow",
    # Tasks
    "load_data",
    "train_model",
    "evaluate_model",
    "check_drift",
    "send_alert",
]