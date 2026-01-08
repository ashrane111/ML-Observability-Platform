"""
Integration Tests for Full Pipeline

End-to-end tests that verify the complete ML pipeline works together.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401
import pytest

from src.data.drift_injector import DriftConfig, DriftInjector, DriftType
from src.data.synthetic_generator import SyntheticDataGenerator
from src.models.churn_predictor import ChurnPredictor
from src.models.fraud_detector import FraudDetector
from src.models.preprocessing import (
    FeaturePreprocessor,
    prepare_churn_features,
    prepare_fraud_features,
    prepare_price_features,
)
from src.models.price_predictor import PricePredictor
from src.monitoring.alerts import AlertManager, AlertSeverity, AlertType
from src.monitoring.drift_detector import DriftDetector, DriftStatus


class TestFraudPipeline:
    """End-to-end tests for fraud detection pipeline."""

    @pytest.fixture
    def fraud_data(self):
        """Generate fraud detection data."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_fraud_data(n_samples=1000)

    @pytest.fixture
    def trained_fraud_model(self, fraud_data):
        """Train a fraud detection model."""
        X_raw, y = prepare_fraud_features(fraud_data)

        # Preprocess features
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(X_raw)

        model = FraudDetector()
        model.fit(X, y)
        return model, preprocessor

    def test_full_fraud_pipeline(self, fraud_data, trained_fraud_model):
        """Test complete fraud detection pipeline."""
        model, preprocessor = trained_fraud_model

        # Split data
        test_data = fraud_data.iloc[800:]

        # Prepare test features
        X_raw, y_test = prepare_fraud_features(test_data)
        X_test = preprocessor.transform(X_raw)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_fraud_probability(X_test)

        # Validate outputs
        assert len(predictions) == len(test_data)
        assert len(probabilities) == len(test_data)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        assert "test_accuracy" in metrics
        assert "test_f1" in metrics
        assert metrics["test_accuracy"] > 0.5  # Better than random

    def test_fraud_model_save_load(self, trained_fraud_model):
        """Test model serialization."""
        model, _ = trained_fraud_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save creates a directory, so we need to find the actual file
            save_dir = Path(tmpdir) / "fraud_model"
            model.save(save_dir)

            # Find the saved pickle file
            pkl_files = list(save_dir.glob("*.pkl"))
            if pkl_files:
                load_path = pkl_files[0]
            else:
                # If no pkl files, try loading from the directory itself
                load_path = save_dir

            loaded_model = FraudDetector.load(load_path)
            assert loaded_model.model_name == model.model_name
            assert loaded_model.version == model.version


class TestPricePipeline:
    """End-to-end tests for price prediction pipeline."""

    @pytest.fixture
    def price_data(self):
        """Generate price prediction data."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_price_data(n_samples=500)

    @pytest.fixture
    def trained_price_model(self, price_data):
        """Train a price prediction model."""
        X_raw, y = prepare_price_features(price_data)

        # Preprocess features
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(X_raw)

        model = PricePredictor()
        model.fit(X, y)
        return model, preprocessor

    def test_full_price_pipeline(self, price_data, trained_price_model):
        """Test complete price prediction pipeline."""
        model, preprocessor = trained_price_model

        # Split data
        test_data = price_data.iloc[400:]

        # Prepare test features
        X_raw, y_test = prepare_price_features(test_data)
        X_test = preprocessor.transform(X_raw)

        # Make predictions
        predictions = model.predict(X_test)

        # Validate outputs
        assert len(predictions) == len(test_data)
        assert all(p > 0 for p in predictions)  # Prices should be positive

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        assert "test_rmse" in metrics
        assert "test_r2" in metrics


class TestChurnPipeline:
    """End-to-end tests for churn prediction pipeline."""

    @pytest.fixture
    def churn_data(self):
        """Generate churn prediction data."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_churn_data(n_samples=800)

    @pytest.fixture
    def trained_churn_model(self, churn_data):
        """Train a churn prediction model."""
        X_raw, y = prepare_churn_features(churn_data)

        # Preprocess features
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(X_raw)

        model = ChurnPredictor()
        model.fit(X, y)
        return model, preprocessor

    def test_full_churn_pipeline(self, churn_data, trained_churn_model):
        """Test complete churn prediction pipeline."""
        model, preprocessor = trained_churn_model

        # Split data
        test_data = churn_data.iloc[600:]

        # Prepare test features
        X_raw, y_test = prepare_churn_features(test_data)
        X_test = preprocessor.transform(X_raw)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_churn_probability(X_test)

        # Validate outputs
        assert len(predictions) == len(test_data)
        assert len(probabilities) == len(test_data)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        assert "test_accuracy" in metrics
        assert "test_f1" in metrics


class TestDriftDetectionPipeline:
    """End-to-end tests for drift detection pipeline."""

    @pytest.fixture
    def reference_and_current_data(self):
        """Generate reference and current datasets."""
        generator = SyntheticDataGenerator(seed=42)
        injector = DriftInjector(seed=42)

        # Generate reference data
        reference = generator.generate_fraud_data(n_samples=1000)

        # Generate current data with drift
        current = generator.generate_fraud_data(n_samples=500)

        # Create drift config with correct parameters
        drift_config = DriftConfig(
            drift_type=DriftType.SUDDEN,
            features=["amount", "distance_from_home"],
            magnitude=0.5,
            affected_fraction=0.5,
        )
        current_drifted = injector.inject_drift(current, drift_config)

        return reference, current_drifted

    def test_drift_detection_pipeline(self, reference_and_current_data):
        """Test complete drift detection pipeline."""
        reference, current = reference_and_current_data

        # Setup detector
        detector = DriftDetector(
            psi_threshold_warning=0.1,
            psi_threshold_critical=0.2,
        )
        detector.set_reference_data(reference, target_column="is_fraud")

        # Detect drift
        result = detector.detect_drift(current, dataset_name="test")

        # Validate result
        assert result is not None
        assert result.dataset_name == "test"
        assert isinstance(result.drift_status, DriftStatus)
        assert 0 <= result.drift_share <= 1
        assert len(result.feature_drift_scores) > 0

    def test_data_quality_check(self, reference_and_current_data):
        """Test data quality checking."""
        reference, current = reference_and_current_data

        # Add some missing values
        current_with_missing = current.copy()
        current_with_missing.loc[:50, "amount"] = np.nan

        # Setup detector
        detector = DriftDetector()
        detector.set_reference_data(reference, target_column="is_fraud")

        # Check quality
        result = detector.check_data_quality(current_with_missing, dataset_name="test")

        # Validate result
        assert result is not None
        assert result.total_rows == len(current_with_missing)
        assert result.missing_values_share > 0
        assert "amount" in result.columns_with_missing


class TestAlertingPipeline:
    """End-to-end tests for alerting pipeline."""

    def test_alert_creation_and_management(self):
        """Test alert creation and lifecycle."""
        manager = AlertManager()

        # Create an alert
        alert = manager.create_alert(
            alert_type=AlertType.DRIFT_DETECTED,
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            title="Test Alert",
            message="This is a test alert",
            metric_name="drift_share",
            metric_value=0.25,
        )

        assert alert is not None
        assert alert.alert_id is not None

        # Get active alerts
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) > 0

        # Acknowledge alert
        acknowledged = manager.acknowledge_alert(alert.alert_id, acknowledged_by="test_user")
        assert acknowledged is not None

        # Resolve alert
        resolved = manager.resolve_alert(
            alert.alert_id,
            resolved_by="test_user",
            resolution_notes="Fixed the issue",
        )
        assert resolved is not None

    def test_drift_alert_creation(self):
        """Test automatic drift alert creation."""
        manager = AlertManager()

        # Create drift alert
        alert = manager.create_drift_alert(
            model_name="fraud_detector",
            drift_share=0.35,
            drifted_features=["amount", "distance_from_home", "hour_of_day"],
            dataset_name="production",
        )

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL  # > 0.3 threshold


class TestFullIntegration:
    """Full integration test combining all components."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        generator = SyntheticDataGenerator(seed=42)
        injector = DriftInjector(seed=42)

        # 1. Generate training data
        train_data = generator.generate_fraud_data(n_samples=1000)

        # 2. Prepare and preprocess features
        X_raw, y_train = prepare_fraud_features(train_data)
        preprocessor = FeaturePreprocessor()
        X_train = preprocessor.fit_transform(X_raw)

        # 3. Train model
        model = FraudDetector()
        model.fit(X_train, y_train)

        # 4. Setup drift detector with reference data
        detector = DriftDetector()
        detector.set_reference_data(train_data, target_column="is_fraud")

        # 5. Setup alert manager
        alert_manager = AlertManager()

        # 6. Simulate production: generate current data with drift
        current_data = generator.generate_fraud_data(n_samples=200)
        drift_config = DriftConfig(
            drift_type=DriftType.GRADUAL,
            features=["amount"],
            magnitude=0.4,
            affected_fraction=0.5,
        )
        current_data_drifted = injector.inject_drift(current_data, drift_config)

        # 7. Make predictions
        X_current_raw, _ = prepare_fraud_features(current_data_drifted)
        X_current = preprocessor.transform(X_current_raw)
        predictions = model.predict(X_current)
        assert len(predictions) == len(current_data_drifted)

        # 8. Check for drift
        drift_result = detector.detect_drift(current_data_drifted, dataset_name="prod")

        # 9. Create alert if drift detected
        if drift_result.drift_share > 0.2:
            alert = alert_manager.create_drift_alert(
                model_name="fraud_detector",
                drift_share=drift_result.drift_share,
                drifted_features=drift_result.drifted_features,
                dataset_name="prod",
            )
            # Alert may or may not be created depending on thresholds
            if alert:
                assert alert.model_name == "fraud_detector"

        # 10. Check data quality
        quality_result = detector.check_data_quality(current_data_drifted, dataset_name="prod")
        assert quality_result.total_rows == len(current_data_drifted)

        # Workflow completed successfully!
