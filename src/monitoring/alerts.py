"""
Alerts Module

Manages alert generation, routing, and history for ML monitoring.
Supports:
- Drift alerts
- Performance degradation alerts
- Data quality alerts
- Custom alert rules
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from loguru import logger


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""

    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    MODEL_ERROR = "model_error"
    LATENCY_HIGH = "latency_high"
    THROUGHPUT_LOW = "throughput_low"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    MISSING_VALUES = "missing_values"
    OUTLIERS_DETECTED = "outliers_detected"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Represents a monitoring alert."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    model_name: str
    title: str
    message: str
    created_at: datetime
    updated_at: datetime

    # Optional fields
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    feature_name: Optional[str] = None
    dataset_name: Optional[str] = None

    # Additional context
    context: dict = field(default_factory=dict)

    # Resolution info
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "model_name": self.model_name,
            "title": self.title,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "feature_name": self.feature_name,
            "dataset_name": self.dataset_name,
            "context": self.context,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
        }

    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Alert":
        """Create alert from dictionary."""
        return cls(
            alert_id=data["alert_id"],
            alert_type=AlertType(data["alert_type"]),
            severity=AlertSeverity(data["severity"]),
            status=AlertStatus(data["status"]),
            model_name=data["model_name"],
            title=data["title"],
            message=data["message"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metric_name=data.get("metric_name"),
            metric_value=data.get("metric_value"),
            threshold=data.get("threshold"),
            feature_name=data.get("feature_name"),
            dataset_name=data.get("dataset_name"),
            context=data.get("context", {}),
            resolved_at=(
                datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None
            ),
            resolved_by=data.get("resolved_by"),
            resolution_notes=data.get("resolution_notes"),
        )


@dataclass
class AlertRule:
    """Defines a rule for generating alerts."""

    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    condition: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    threshold: float
    enabled: bool = True
    cooldown_minutes: int = 30  # Minimum time between alerts
    description: Optional[str] = None
    model_name: Optional[str] = None  # None = applies to all models

    def evaluate(self, metric_value: float) -> bool:
        """
        Evaluate if the rule triggers.

        Args:
            metric_value: Current metric value

        Returns:
            True if alert should be triggered
        """
        if not self.enabled:
            return False

        conditions = {
            "gt": metric_value > self.threshold,
            "lt": metric_value < self.threshold,
            "gte": metric_value >= self.threshold,
            "lte": metric_value <= self.threshold,
            "eq": metric_value == self.threshold,
        }

        return conditions.get(self.condition, False)


class AlertManager:
    """
    Manages alert lifecycle and routing.

    Handles:
    - Alert creation and deduplication
    - Alert status management
    - Alert history
    - Rule evaluation
    """

    def __init__(self) -> None:
        """Initialize the alert manager."""
        self._alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._rules: dict[str, AlertRule] = {}
        self._last_alert_time: dict[str, datetime] = {}

        # Register default rules
        self._register_default_rules()

        logger.info("AlertManager initialized")

    def _register_default_rules(self) -> None:
        """Register default alerting rules."""
        default_rules = [
            AlertRule(
                rule_id="drift_warning",
                name="Drift Warning",
                alert_type=AlertType.DRIFT_DETECTED,
                severity=AlertSeverity.WARNING,
                metric_name="drift_share",
                condition="gt",
                threshold=0.2,
                description="More than 20% of features show drift",
            ),
            AlertRule(
                rule_id="drift_critical",
                name="Drift Critical",
                alert_type=AlertType.DRIFT_DETECTED,
                severity=AlertSeverity.CRITICAL,
                metric_name="drift_share",
                condition="gt",
                threshold=0.3,
                description="More than 30% of features show drift",
            ),
            AlertRule(
                rule_id="psi_warning",
                name="PSI Warning",
                alert_type=AlertType.FEATURE_DRIFT,
                severity=AlertSeverity.WARNING,
                metric_name="psi_score",
                condition="gt",
                threshold=0.1,
                description="Feature PSI exceeds warning threshold",
            ),
            AlertRule(
                rule_id="psi_critical",
                name="PSI Critical",
                alert_type=AlertType.FEATURE_DRIFT,
                severity=AlertSeverity.CRITICAL,
                metric_name="psi_score",
                condition="gt",
                threshold=0.2,
                description="Feature PSI exceeds critical threshold",
            ),
            AlertRule(
                rule_id="accuracy_degradation",
                name="Accuracy Degradation",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.WARNING,
                metric_name="accuracy",
                condition="lt",
                threshold=0.85,
                description="Model accuracy dropped below threshold",
            ),
            AlertRule(
                rule_id="f1_degradation",
                name="F1 Score Degradation",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.WARNING,
                metric_name="f1_score",
                condition="lt",
                threshold=0.7,
                description="Model F1 score dropped below threshold",
            ),
            AlertRule(
                rule_id="missing_values_warning",
                name="Missing Values Warning",
                alert_type=AlertType.MISSING_VALUES,
                severity=AlertSeverity.WARNING,
                metric_name="missing_values_share",
                condition="gt",
                threshold=0.05,
                description="More than 5% missing values detected",
            ),
            AlertRule(
                rule_id="latency_warning",
                name="High Latency Warning",
                alert_type=AlertType.LATENCY_HIGH,
                severity=AlertSeverity.WARNING,
                metric_name="p99_latency_ms",
                condition="gt",
                threshold=200,
                description="P99 prediction latency exceeds 200ms",
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

        logger.debug(f"Registered {len(default_rules)} default alert rules")

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        model_name: str,
        title: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        feature_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> Alert:
        """
        Create a new alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            model_name: Name of the model
            title: Alert title
            message: Alert message
            metric_name: Name of the metric that triggered the alert
            metric_value: Current value of the metric
            threshold: Threshold that was exceeded
            feature_name: Feature name (for feature-specific alerts)
            dataset_name: Dataset name
            context: Additional context

        Returns:
            Created Alert object
        """
        now = datetime.now()

        alert = Alert(
            alert_id=str(uuid4()),
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            model_name=model_name,
            title=title,
            message=message,
            created_at=now,
            updated_at=now,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            feature_name=feature_name,
            dataset_name=dataset_name,
            context=context or {},
        )

        self._alerts[alert.alert_id] = alert
        self._alert_history.append(alert)

        logger.warning(
            f"Alert created: [{severity.value.upper()}] {title} "
            f"(model={model_name}, type={alert_type.value})"
        )

        return alert

    def evaluate_rules(
        self,
        model_name: str,
        metrics: dict[str, float],
        dataset_name: Optional[str] = None,
    ) -> list[Alert]:
        """
        Evaluate all rules against provided metrics.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metric names to values
            dataset_name: Optional dataset name

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self._rules.values():
            # Skip if rule is for a specific model that doesn't match
            if rule.model_name and rule.model_name != model_name:
                continue

            # Check if metric exists
            if rule.metric_name not in metrics:
                continue

            metric_value = metrics[rule.metric_name]

            # Evaluate rule
            if rule.evaluate(metric_value):
                # Check cooldown
                cooldown_key = f"{model_name}:{rule.rule_id}"
                last_time = self._last_alert_time.get(cooldown_key)

                if last_time:
                    elapsed = (datetime.now() - last_time).total_seconds() / 60
                    if elapsed < rule.cooldown_minutes:
                        logger.debug(
                            f"Skipping alert {rule.rule_id} due to cooldown "
                            f"({elapsed:.1f} < {rule.cooldown_minutes} minutes)"
                        )
                        continue

                # Create alert
                alert = self.create_alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    model_name=model_name,
                    title=rule.name,
                    message=rule.description
                    or f"{rule.metric_name} {rule.condition} {rule.threshold}",
                    metric_name=rule.metric_name,
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    dataset_name=dataset_name,
                )

                triggered_alerts.append(alert)
                self._last_alert_time[cooldown_key] = datetime.now()

        return triggered_alerts

    def create_drift_alert(
        self,
        model_name: str,
        drift_share: float,
        drifted_features: list[str],
        dataset_name: str = "production",
        feature_scores: Optional[dict[str, float]] = None,
    ) -> Optional[Alert]:
        """
        Create an alert for detected drift.

        Args:
            model_name: Name of the model
            drift_share: Fraction of features with drift
            drifted_features: List of drifted feature names
            dataset_name: Name of the dataset
            feature_scores: Per-feature drift scores

        Returns:
            Created Alert or None if no alert needed
        """
        # Determine severity
        if drift_share > 0.3:
            severity = AlertSeverity.CRITICAL
        elif drift_share > 0.2:
            severity = AlertSeverity.WARNING
        else:
            return None

        title = f"Data Drift Detected - {model_name}"
        message = (
            f"Drift detected in {len(drifted_features)} features "
            f"({drift_share:.1%} of total). "
            f"Affected features: {', '.join(drifted_features[:5])}"
        )
        if len(drifted_features) > 5:
            message += f" and {len(drifted_features) - 5} more"

        return self.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=severity,
            model_name=model_name,
            title=title,
            message=message,
            metric_name="drift_share",
            metric_value=drift_share,
            threshold=0.2 if severity == AlertSeverity.WARNING else 0.3,
            dataset_name=dataset_name,
            context={
                "drifted_features": drifted_features,
                "feature_scores": feature_scores or {},
            },
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Who acknowledged the alert

        Returns:
            Updated Alert or None if not found
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert {alert_id} not found")
            return None

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.updated_at = datetime.now()

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return alert

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Resolve an alert.

        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
            resolution_notes: Notes about the resolution

        Returns:
            Updated Alert or None if not found
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert {alert_id} not found")
            return None

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = alert.resolved_at
        alert.resolved_by = resolved_by
        alert.resolution_notes = resolution_notes

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return alert

    def get_active_alerts(
        self,
        model_name: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            model_name: Filter by model name
            alert_type: Filter by alert type
            severity: Filter by severity

        Returns:
            List of matching active alerts
        """
        alerts = [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[Alert]:
        """
        Get alert history.

        Args:
            model_name: Filter by model name
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        alerts = self._alert_history

        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:limit]

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by its ID."""
        return self._alerts.get(alert_id)

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self._rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def get_rules(self) -> list[AlertRule]:
        """Get all alert rules."""
        return list(self._rules.values())

    def get_alert_summary(self, model_name: Optional[str] = None) -> dict[str, Any]:
        """
        Get a summary of current alert status.

        Args:
            model_name: Filter by model name

        Returns:
            Summary dictionary
        """
        active = self.get_active_alerts(model_name=model_name)

        return {
            "total_active": len(active),
            "by_severity": {
                "critical": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in active if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in active if a.severity == AlertSeverity.INFO]),
            },
            "by_type": {
                t.value: len([a for a in active if a.alert_type == t])
                for t in AlertType
                if any(a.alert_type == t for a in active)
            },
            "oldest_alert": (active[-1].created_at.isoformat() if active else None),
            "newest_alert": (active[0].created_at.isoformat() if active else None),
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Get the global AlertManager instance.

    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
