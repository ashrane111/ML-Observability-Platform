"""Prefect workflows for ML Observability Platform.

Provides automated workflows for:
- Model retraining based on drift detection
- Data quality monitoring
- Scheduled model evaluation
- Alert-triggered remediation
"""

from .monitoring import data_quality_flow, drift_monitoring_flow, model_health_flow
from .retraining import drift_triggered_retraining_flow, retraining_flow, scheduled_retraining_flow
from .tasks import check_drift, evaluate_model, load_data, send_alert, train_model

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
