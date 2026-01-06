"""
Unit Tests for Monitoring Module

Tests for drift detection, metrics, and alerts.
"""

# from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity, AlertStatus, AlertType
from src.monitoring.drift_detector import DriftDetector, DriftStatus, create_drift_detector
from src.monitoring.metrics import MetricsCollector, get_metrics


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(100, 10, 1000),
                "feature2": np.random.normal(50, 5, 1000),
                "feature3": np.random.choice(["a", "b", "c"], 1000),
                "target": np.random.choice([0, 1], 1000),
            }
        )

    @pytest.fixture
    def current_data_no_drift(self, reference_data):
        """Create current dataset without drift."""
        np.random.seed(43)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(100, 10, 500),
                "feature2": np.random.normal(50, 5, 500),
                "feature3": np.random.choice(["a", "b", "c"], 500),
                "target": np.random.choice([0, 1], 500),
            }
        )

    @pytest.fixture
    def current_data_with_drift(self, reference_data):
        """Create current dataset with drift."""
        np.random.seed(44)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(150, 15, 500),  # Shifted mean
                "feature2": np.random.normal(80, 10, 500),  # Shifted mean
                "feature3": np.random.choice(
                    ["a", "b", "c"], 500, p=[0.8, 0.1, 0.1]
                ),  # Changed distribution
                "target": np.random.choice([0, 1], 500),
            }
        )

    def test_detector_initialization(self):
        """Test drift detector initializes correctly."""
        detector = DriftDetector(
            psi_threshold_warning=0.1,
            psi_threshold_critical=0.2,
        )

        assert detector.psi_threshold_warning == 0.1
        assert detector.psi_threshold_critical == 0.2

    def test_set_reference_data(self, reference_data):
        """Test setting reference data."""
        detector = DriftDetector()
        detector.set_reference_data(reference_data, target_column="target")

        assert detector._reference_data is not None
        assert len(detector._reference_data) == len(reference_data)
        assert len(detector._numerical_features) > 0
        assert len(detector._categorical_features) > 0

    def test_detect_drift_no_drift(self, reference_data, current_data_no_drift):
        """Test drift detection with no drift."""
        detector = DriftDetector()
        detector.set_reference_data(reference_data, target_column="target")

        result = detector.detect_drift(current_data_no_drift, dataset_name="test")

        assert result is not None
        assert result.dataset_name == "test"
        assert isinstance(result.drift_status, DriftStatus)
        assert isinstance(result.drift_share, float)

    def test_detect_drift_with_drift(self, reference_data, current_data_with_drift):
        """Test drift detection with significant drift."""
        detector = DriftDetector()
        detector.set_reference_data(reference_data, target_column="target")

        result = detector.detect_drift(current_data_with_drift, dataset_name="test")

        assert result is not None
        # With significant drift, we expect detection
        assert result.drift_share > 0 or len(result.drifted_features) > 0

    def test_detect_drift_without_reference_raises_error(self):
        """Test that drift detection without reference data raises error."""
        detector = DriftDetector()
        current = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError, match="Reference data not set"):
            detector.detect_drift(current)

    def test_detect_feature_drift(self, reference_data, current_data_with_drift):
        """Test single feature drift detection."""
        detector = DriftDetector()
        detector.set_reference_data(reference_data, target_column="target")

        result = detector.detect_feature_drift(current_data_with_drift, "feature1")

        assert "feature" in result
        assert result["feature"] == "feature1"
        assert "drift_detected" in result
        assert "drift_score" in result

    def test_check_data_quality(self, reference_data):
        """Test data quality check."""
        detector = DriftDetector()
        detector.set_reference_data(reference_data, target_column="target")

        # Add some missing values
        data_with_issues = reference_data.copy()
        data_with_issues.loc[0:10, "feature1"] = np.nan

        result = detector.check_data_quality(data_with_issues, dataset_name="test")

        assert result is not None
        assert result.dataset_name == "test"
        assert result.total_rows == len(data_with_issues)

    def test_drift_result_to_dict(self, reference_data, current_data_no_drift):
        """Test DriftResult serialization."""
        detector = DriftDetector()
        detector.set_reference_data(reference_data, target_column="target")

        result = detector.detect_drift(current_data_no_drift)
        result_dict = result.to_dict()

        assert "timestamp" in result_dict
        assert "drift_status" in result_dict
        assert "drift_share" in result_dict
        assert "drifted_features" in result_dict

    def test_create_drift_detector_convenience(self, reference_data):
        """Test convenience function for creating detector."""
        detector = create_drift_detector(reference_data, target_column="target")

        assert detector is not None
        assert detector._reference_data is not None


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a metrics collector."""
        return MetricsCollector(model_name="test_model", model_version="1.0.0")

    def test_collector_initialization(self, collector):
        """Test collector initializes correctly."""
        assert collector.model_name == "test_model"
        assert collector.model_version == "1.0.0"

    def test_record_prediction(self, collector):
        """Test recording prediction metrics."""
        # Should not raise
        collector.record_prediction(
            latency_seconds=0.05,
            prediction_value=0.75,
            batch_size=10,
            status="success",
        )

    def test_record_drift(self, collector):
        """Test recording drift metrics."""
        collector.record_drift(
            dataset_name="production",
            drift_detected=True,
            drift_share=0.25,
            drifted_features_count=3,
            feature_scores={"feature1": 0.15, "feature2": 0.3},
        )

    def test_record_performance(self, collector):
        """Test recording performance metrics."""
        metrics = {
            "test_accuracy": 0.95,
            "test_f1": 0.92,
            "test_precision": 0.93,
            "test_recall": 0.91,
        }
        collector.record_performance(metrics, dataset="test")

    def test_record_data_quality(self, collector):
        """Test recording data quality metrics."""
        collector.record_data_quality(
            dataset_name="production",
            missing_share=0.02,
            duplicate_count=5,
            quality_score=0.95,
        )

    def test_record_alert(self, collector):
        """Test recording alert metrics."""
        collector.record_alert(alert_type="drift", severity="warning")

    def test_set_model_loaded(self, collector):
        """Test setting model loaded status."""
        collector.set_model_loaded(True)
        collector.set_model_loaded(False)

    def test_get_metrics(self):
        """Test getting Prometheus metrics."""
        metrics_output = get_metrics()

        assert metrics_output is not None
        assert isinstance(metrics_output, bytes)


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.fixture
    def alert_manager(self):
        """Create an alert manager."""
        return AlertManager()

    def test_manager_initialization(self, alert_manager):
        """Test alert manager initializes correctly."""
        assert alert_manager is not None
        # Should have default rules
        rules = alert_manager.get_rules()
        assert len(rules) > 0

    def test_create_alert(self, alert_manager):
        """Test creating an alert."""
        alert = alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DRIFT_DETECTED
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.ACTIVE
        assert alert.model_name == "test_model"

    def test_create_drift_alert(self, alert_manager):
        """Test creating a drift-specific alert."""
        alert = alert_manager.create_drift_alert(
            model_name="test_model",
            drift_share=0.35,
            drifted_features=["feature1", "feature2", "feature3"],
            dataset_name="production",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DRIFT_DETECTED
        assert alert.severity == AlertSeverity.CRITICAL  # > 0.3 threshold

    def test_create_drift_alert_warning(self, alert_manager):
        """Test creating a warning-level drift alert."""
        alert = alert_manager.create_drift_alert(
            model_name="test_model",
            drift_share=0.25,
            drifted_features=["feature1"],
            dataset_name="production",
        )

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING  # Between 0.2 and 0.3

    def test_no_alert_for_low_drift(self, alert_manager):
        """Test that no alert is created for low drift."""
        alert = alert_manager.create_drift_alert(
            model_name="test_model",
            drift_share=0.1,
            drifted_features=[],
            dataset_name="production",
        )

        assert alert is None

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        alert = alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test",
            message="Test",
        )

        updated = alert_manager.acknowledge_alert(alert.alert_id, acknowledged_by="user1")

        assert updated is not None
        assert updated.status == AlertStatus.ACKNOWLEDGED

    def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        alert = alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test",
            message="Test",
        )

        resolved = alert_manager.resolve_alert(
            alert.alert_id,
            resolved_by="user1",
            resolution_notes="Fixed the issue",
        )

        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED
        assert resolved.resolved_by == "user1"
        assert resolved.resolution_notes == "Fixed the issue"

    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Create some alerts
        alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="model1",
            title="Alert 1",
            message="Message 1",
        )
        alert_manager.create_alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.CRITICAL,
            model_name="model2",
            title="Alert 2",
            message="Message 2",
        )

        # Get all active
        all_active = alert_manager.get_active_alerts()
        assert len(all_active) >= 2

        # Filter by model
        model1_alerts = alert_manager.get_active_alerts(model_name="model1")
        assert all(a.model_name == "model1" for a in model1_alerts)

        # Filter by severity
        critical_alerts = alert_manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical_alerts)

    def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        # Create an alert
        alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test",
            message="Test",
        )

        history = alert_manager.get_alert_history()
        assert len(history) > 0

    def test_evaluate_rules(self, alert_manager):
        """Test rule evaluation."""
        metrics = {
            "drift_share": 0.35,  # Should trigger critical drift alert
            "accuracy": 0.95,  # Should not trigger
        }

        alerts = alert_manager.evaluate_rules(
            model_name="test_model",
            metrics=metrics,
        )

        # Should have at least one alert for drift_share
        drift_alerts = [a for a in alerts if a.alert_type == AlertType.DRIFT_DETECTED]
        assert len(drift_alerts) > 0

    def test_add_custom_rule(self, alert_manager):
        """Test adding a custom rule."""
        rule = AlertRule(
            rule_id="custom_rule",
            name="Custom Rule",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.WARNING,
            metric_name="custom_metric",
            condition="lt",
            threshold=0.5,
        )

        alert_manager.add_rule(rule)

        rules = alert_manager.get_rules()
        rule_ids = [r.rule_id for r in rules]
        assert "custom_rule" in rule_ids

    def test_remove_rule(self, alert_manager):
        """Test removing a rule."""
        # Add then remove
        rule = AlertRule(
            rule_id="to_remove",
            name="To Remove",
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.INFO,
            metric_name="test",
            condition="gt",
            threshold=0.5,
        )
        alert_manager.add_rule(rule)

        result = alert_manager.remove_rule("to_remove")
        assert result is True

        rules = alert_manager.get_rules()
        rule_ids = [r.rule_id for r in rules]
        assert "to_remove" not in rule_ids

    def test_get_alert_summary(self, alert_manager):
        """Test getting alert summary."""
        # Create some alerts
        alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test",
            message="Test",
        )

        summary = alert_manager.get_alert_summary()

        assert "total_active" in summary
        assert "by_severity" in summary
        assert "by_type" in summary

    def test_alert_to_dict(self, alert_manager):
        """Test alert serialization."""
        alert = alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test",
            message="Test",
            metric_name="drift_share",
            metric_value=0.25,
        )

        alert_dict = alert.to_dict()

        assert "alert_id" in alert_dict
        assert "alert_type" in alert_dict
        assert "severity" in alert_dict
        assert alert_dict["metric_value"] == 0.25

    def test_alert_to_json(self, alert_manager):
        """Test alert JSON serialization."""
        alert = alert_manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test",
            message="Test",
        )

        json_str = alert.to_json()
        assert isinstance(json_str, str)
        assert "alert_id" in json_str


class TestAlertRule:
    """Tests for AlertRule class."""

    def test_rule_evaluate_gt(self):
        """Test greater than condition."""
        rule = AlertRule(
            rule_id="test",
            name="Test",
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            metric_name="test",
            condition="gt",
            threshold=0.5,
        )

        assert rule.evaluate(0.6) is True
        assert rule.evaluate(0.5) is False
        assert rule.evaluate(0.4) is False

    def test_rule_evaluate_lt(self):
        """Test less than condition."""
        rule = AlertRule(
            rule_id="test",
            name="Test",
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            metric_name="test",
            condition="lt",
            threshold=0.5,
        )

        assert rule.evaluate(0.4) is True
        assert rule.evaluate(0.5) is False
        assert rule.evaluate(0.6) is False

    def test_rule_evaluate_disabled(self):
        """Test disabled rule."""
        rule = AlertRule(
            rule_id="test",
            name="Test",
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            metric_name="test",
            condition="gt",
            threshold=0.5,
            enabled=False,
        )

        assert rule.evaluate(0.6) is False
