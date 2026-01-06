"""
Unit Tests for Models Module

Tests for all ML models and preprocessing.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.churn_predictor import ChurnPredictor
from src.models.fraud_detector import FraudDetector
from src.models.preprocessing import (
    FeaturePreprocessor,
    prepare_churn_features,
    prepare_fraud_features,
    prepare_price_features,
)
from src.models.price_predictor import PricePredictor


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with mixed types."""
        return pd.DataFrame(
            {
                "num_feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "num_feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "cat_feature": ["a", "b", "a", "c", "b"],
                "bool_feature": [True, False, True, False, True],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = FeaturePreprocessor(
            numerical_strategy="standard",
            categorical_strategy="onehot",
        )
        assert preprocessor.numerical_strategy == "standard"
        assert preprocessor.categorical_strategy == "onehot"
        assert not preprocessor.is_fitted

    def test_fit_identifies_feature_types(self, sample_df):
        """Test that fit correctly identifies feature types."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_df, target_col="target")

        assert "num_feature1" in preprocessor.numerical_features
        assert "num_feature2" in preprocessor.numerical_features
        assert "cat_feature" in preprocessor.categorical_features
        assert "bool_feature" in preprocessor.boolean_features
        assert "target" not in preprocessor.numerical_features

    def test_transform_returns_dataframe(self, sample_df):
        """Test that transform returns a DataFrame."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_df, target_col="target")
        result = preprocessor.transform(sample_df, target_col="target")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_fit_transform(self, sample_df):
        """Test fit_transform convenience method."""
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(sample_df, target_col="target")

        assert preprocessor.is_fitted
        assert isinstance(result, pd.DataFrame)

    def test_transform_without_fit_raises_error(self, sample_df):
        """Test that transform raises error if not fitted."""
        preprocessor = FeaturePreprocessor()
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(sample_df)

    def test_get_feature_info(self, sample_df):
        """Test get_feature_info returns correct info."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_df, target_col="target")
        info = preprocessor.get_feature_info()

        assert "numerical_features" in info
        assert "categorical_features" in info
        assert "boolean_features" in info
        assert info["is_fitted"] is True


class TestFraudDetector:
    """Tests for FraudDetector model."""

    @pytest.fixture
    def fraud_data(self):
        """Create sample fraud detection data."""
        np.random.seed(42)
        n_samples = 500
        n_fraud = 25

        X = pd.DataFrame(
            {
                "amount": np.random.lognormal(3, 1, n_samples),
                "distance_from_home": np.random.exponential(10, n_samples),
                "hour_of_day": np.random.randint(0, 24, n_samples),
                "transaction_count_24h": np.random.poisson(3, n_samples),
            }
        )

        y = pd.Series([False] * (n_samples - n_fraud) + [True] * n_fraud)
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

        return X, y

    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = FraudDetector(version="1.0.0")

        assert model.model_name == "fraud_detector"
        assert model.model_type == "classification"
        assert model.version == "1.0.0"
        assert not model.is_fitted

    def test_model_fit(self, fraud_data):
        """Test model training."""
        X, y = fraud_data
        model = FraudDetector()
        model.fit(X, y)

        assert model.is_fitted
        assert model.training_date is not None
        assert len(model.feature_names) == X.shape[1]

    def test_model_predict(self, fraud_data):
        """Test model prediction."""
        X, y = fraud_data
        model = FraudDetector()
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        # Check predictions are binary (0/1 or True/False)
        unique_preds = set(predictions)
        assert unique_preds <= {0, 1} or unique_preds <= {True, False}

    def test_predict_proba(self, fraud_data):
        """Test probability prediction."""
        X, y = fraud_data
        model = FraudDetector()
        model.fit(X, y)

        proba = model.predict_proba(X)

        assert proba is not None
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_fraud_probability(self, fraud_data):
        """Test fraud probability method."""
        X, y = fraud_data
        model = FraudDetector()
        model.fit(X, y)

        fraud_proba = model.predict_fraud_probability(X)

        assert len(fraud_proba) == len(X)
        assert all(0 <= p <= 1 for p in fraud_proba)

    def test_evaluate_returns_metrics(self, fraud_data):
        """Test evaluation returns expected metrics."""
        X, y = fraud_data
        model = FraudDetector()
        model.fit(X, y)

        metrics = model.evaluate(X, y, prefix="test")

        assert "test_accuracy" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "test_f1" in metrics

    def test_feature_importance(self, fraud_data):
        """Test feature importance extraction."""
        X, y = fraud_data
        model = FraudDetector()
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())

    def test_predict_without_fit_raises_error(self, fraud_data):
        """Test prediction without training raises error."""
        X, _ = fraud_data
        model = FraudDetector()

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)


class TestPricePredictor:
    """Tests for PricePredictor model."""

    @pytest.fixture
    def price_data(self):
        """Create sample price prediction data."""
        np.random.seed(42)
        n_samples = 300

        X = pd.DataFrame(
            {
                "square_feet": np.random.normal(2000, 500, n_samples),
                "bedrooms": np.random.randint(1, 6, n_samples),
                "bathrooms": np.random.randint(1, 4, n_samples),
                "year_built": np.random.randint(1960, 2023, n_samples),
            }
        )

        # Price correlated with features
        y = (
            X["square_feet"] * 150
            + X["bedrooms"] * 20000
            + X["bathrooms"] * 15000
            + np.random.normal(0, 30000, n_samples)
        )
        y = pd.Series(y.clip(50000, 2000000))

        return X, y

    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = PricePredictor(version="1.0.0")

        assert model.model_name == "price_predictor"
        assert model.model_type == "regression"
        assert model.version == "1.0.0"

    def test_model_fit(self, price_data):
        """Test model training."""
        X, y = price_data
        model = PricePredictor()
        model.fit(X, y)

        assert model.is_fitted

    def test_model_predict(self, price_data):
        """Test model prediction."""
        X, y = price_data
        model = PricePredictor()
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert all(p > 0 for p in predictions)

    def test_evaluate_returns_regression_metrics(self, price_data):
        """Test evaluation returns regression metrics."""
        X, y = price_data
        model = PricePredictor()
        model.fit(X, y)

        metrics = model.evaluate(X, y, prefix="test")

        assert "test_mse" in metrics
        assert "test_rmse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        assert "test_mape" in metrics

    def test_predict_price_range(self, price_data):
        """Test price range prediction."""
        X, y = price_data
        model = PricePredictor()
        model.fit(X, y)

        result = model.predict_price_range(X)

        assert "predicted_price" in result.columns
        assert "price_low" in result.columns
        assert "price_high" in result.columns
        assert all(result["price_low"] <= result["predicted_price"])
        assert all(result["predicted_price"] <= result["price_high"])


class TestChurnPredictor:
    """Tests for ChurnPredictor model."""

    @pytest.fixture
    def churn_data(self):
        """Create sample churn prediction data."""
        np.random.seed(42)
        n_samples = 400
        n_churned = 60

        X = pd.DataFrame(
            {
                "tenure_months": np.random.randint(1, 60, n_samples),
                "monthly_charges": np.random.uniform(20, 100, n_samples),
                "support_tickets": np.random.poisson(2, n_samples),
                "login_frequency": np.random.gamma(3, 1, n_samples),
            }
        )

        y = pd.Series([False] * (n_samples - n_churned) + [True] * n_churned)
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

        return X, y

    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = ChurnPredictor(version="1.0.0")

        assert model.model_name == "churn_predictor"
        assert model.model_type == "classification"
        assert model.version == "1.0.0"

    def test_model_fit(self, churn_data):
        """Test model training."""
        X, y = churn_data
        model = ChurnPredictor()
        model.fit(X, y)

        assert model.is_fitted

    def test_predict_churn_probability(self, churn_data):
        """Test churn probability method."""
        X, y = churn_data
        model = ChurnPredictor()
        model.fit(X, y)

        churn_proba = model.predict_churn_probability(X)

        assert len(churn_proba) == len(X)
        assert all(0 <= p <= 1 for p in churn_proba)

    def test_get_at_risk_customers(self, churn_data):
        """Test at-risk customer identification."""
        X, y = churn_data
        model = ChurnPredictor()
        model.fit(X, y)

        at_risk = model.get_at_risk_customers(X, threshold=0.3)

        assert "churn_probability" in at_risk.columns
        assert "risk_level" in at_risk.columns
        assert all(at_risk["churn_probability"] >= 0.3)

    def test_segment_customers(self, churn_data):
        """Test customer segmentation."""
        X, y = churn_data
        model = ChurnPredictor()
        model.fit(X, y)

        segments = model.segment_customers(X)

        assert "segment" in segments.columns
        assert "churn_probability" in segments.columns

    def test_evaluate_returns_churn_metrics(self, churn_data):
        """Test evaluation returns churn-specific metrics."""
        X, y = churn_data
        model = ChurnPredictor()
        model.fit(X, y)

        metrics = model.evaluate(X, y, prefix="test")

        assert "test_accuracy" in metrics
        assert "test_f1" in metrics


class TestPrepareFeatures:
    """Tests for feature preparation functions."""

    def test_prepare_fraud_features(self, sample_fraud_data):
        """Test fraud feature preparation."""
        X, y = prepare_fraud_features(sample_fraud_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "amount" in X.columns
        assert "is_fraud" not in X.columns

    def test_prepare_price_features(self, sample_price_data):
        """Test price feature preparation."""
        X, y = prepare_price_features(sample_price_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "square_feet" in X.columns
        assert "price" not in X.columns

    def test_prepare_churn_features(self, sample_churn_data):
        """Test churn feature preparation."""
        X, y = prepare_churn_features(sample_churn_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "tenure_months" in X.columns
        assert "churned" not in X.columns


class TestModelSaveLoad:
    """Tests for model serialization."""

    def test_save_and_load_fraud_detector(self, tmp_path):
        """Test saving and loading fraud detector."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.choice([True, False], 100))

        # Train and save
        model = FraudDetector(version="1.0.0")
        model.fit(X, y)
        model_path = model.save(tmp_path)

        # Load and verify
        loaded_model = FraudDetector.load(model_path)

        assert loaded_model.is_fitted
        assert loaded_model.model_name == "fraud_detector"
        assert loaded_model.version == "1.0.0"

        # Predictions should match
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_save_and_load_price_predictor(self, tmp_path):
        """Test saving and loading price predictor."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.uniform(100000, 500000, 100))

        # Train and save
        model = PricePredictor(version="1.0.0")
        model.fit(X, y)
        model_path = model.save(tmp_path)

        # Load and verify
        loaded_model = PricePredictor.load(model_path)

        assert loaded_model.is_fitted
        assert loaded_model.model_name == "price_predictor"

        # Predictions should match
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
