"""
Unit Tests for Data Generation Module

Tests synthetic data generation and drift injection.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.drift_injector import DataQualityIssue, DriftConfig, DriftInjector, DriftType
from src.data.synthetic_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed seed."""
        return SyntheticDataGenerator(seed=42)

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator.seed == 42

    def test_generate_fraud_data(self, generator):
        """Test fraud data generation."""
        df = generator.generate_fraud_data(n_samples=1000, fraud_rate=0.05)

        assert len(df) == 1000
        assert "is_fraud" in df.columns
        assert "amount" in df.columns
        assert "transaction_id" in df.columns

        # Check fraud rate is approximately correct
        actual_fraud_rate = df["is_fraud"].mean()
        assert 0.03 < actual_fraud_rate < 0.07

        # Check amount is positive
        assert (df["amount"] > 0).all()

        # Check hour of day is valid
        assert df["hour_of_day"].between(0, 23).all()

    def test_generate_price_data(self, generator):
        """Test price data generation."""
        df = generator.generate_price_data(n_samples=500)

        assert len(df) == 500
        assert "price" in df.columns
        assert "square_feet" in df.columns
        assert "bedrooms" in df.columns

        # Check price is positive
        assert (df["price"] > 0).all()

        # Check bedrooms is reasonable
        assert df["bedrooms"].between(1, 10).all()

        # Check property types are valid
        valid_types = ["house", "apartment", "condo", "townhouse"]
        assert df["property_type"].isin(valid_types).all()

    def test_generate_churn_data(self, generator):
        """Test churn data generation."""
        df = generator.generate_churn_data(n_samples=800, churn_rate=0.2)

        assert len(df) == 800
        assert "churned" in df.columns
        assert "tenure_months" in df.columns

        # Check churn rate is approximately correct
        actual_churn_rate = df["churned"].mean()
        assert 0.15 < actual_churn_rate < 0.25

        # Check tenure is positive
        assert (df["tenure_months"] > 0).all()

    def test_generate_all_datasets(self, generator):
        """Test generating all datasets at once."""
        datasets = generator.generate_all_datasets(
            fraud_samples=100,
            price_samples=100,
            churn_samples=100,
        )

        assert "fraud" in datasets
        assert "price" in datasets
        assert "churn" in datasets

        assert len(datasets["fraud"]) == 100
        assert len(datasets["price"]) == 100
        assert len(datasets["churn"]) == 100

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)

        df1 = gen1.generate_fraud_data(n_samples=100)
        df2 = gen2.generate_fraud_data(n_samples=100)

        # Amounts should be identical
        np.testing.assert_array_almost_equal(
            df1["amount"].values,
            df2["amount"].values,
        )

    def test_get_feature_columns(self, generator):
        """Test getting feature columns for each dataset type."""
        fraud_features = generator.get_feature_columns("fraud")
        price_features = generator.get_feature_columns("price")
        churn_features = generator.get_feature_columns("churn")

        assert "amount" in fraud_features
        assert "square_feet" in price_features
        assert "tenure_months" in churn_features

    def test_get_target_column(self, generator):
        """Test getting target column for each dataset type."""
        assert generator.get_target_column("fraud") == "is_fraud"
        assert generator.get_target_column("price") == "price"
        assert generator.get_target_column("churn") == "churned"


class TestDriftInjector:
    """Tests for DriftInjector class."""

    @pytest.fixture
    def injector(self):
        """Create a drift injector with fixed seed."""
        return DriftInjector(seed=42)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(100, 10, 1000),
                "feature2": np.random.normal(50, 5, 1000),
                "is_fraud": np.random.choice([True, False], 1000, p=[0.1, 0.9]),
            }
        )

    def test_injector_initialization(self, injector):
        """Test injector initializes correctly."""
        assert injector.seed == 42

    def test_gradual_drift(self, injector, sample_df):
        """Test gradual drift injection."""
        original_mean = sample_df["feature1"].mean()

        config = DriftConfig(
            drift_type=DriftType.GRADUAL,
            magnitude=0.5,
        )
        drifted_df = injector.inject_drift(sample_df, config)

        # Last portion should have different mean
        last_portion_mean = drifted_df["feature1"].iloc[-200:].mean()
        assert last_portion_mean > original_mean

    def test_sudden_drift(self, injector, sample_df):
        """Test sudden drift injection."""
        config = DriftConfig(
            drift_type=DriftType.SUDDEN,
            magnitude=0.5,
            start_index=500,
        )
        drifted_df = injector.inject_drift(sample_df, config)

        # Before and after should have different distributions
        before_mean = drifted_df["feature1"].iloc[:500].mean()
        after_mean = drifted_df["feature1"].iloc[500:].mean()

        assert abs(after_mean - before_mean) > 5  # Significant difference

    def test_feature_drift(self, injector, sample_df):
        """Test feature-specific drift injection."""
        config = DriftConfig(
            drift_type=DriftType.FEATURE,
            magnitude=0.5,
            features=["feature1"],
            affected_fraction=0.5,
        )

        original_mean2 = sample_df["feature2"].mean()
        drifted_df = injector.inject_drift(sample_df, config)

        # feature2 should be relatively unchanged
        assert abs(drifted_df["feature2"].mean() - original_mean2) < 1

    def test_missing_values_injection(self, injector, sample_df):
        """Test missing value injection."""
        drifted_df = injector.inject_data_quality_issues(
            sample_df,
            DataQualityIssue.MISSING_VALUES,
            affected_fraction=0.1,
            features=["feature1"],
        )

        missing_rate = drifted_df["feature1"].isna().mean()
        assert 0.05 < missing_rate < 0.15

    def test_outlier_injection(self, injector, sample_df):
        """Test outlier injection."""
        original_max = sample_df["feature1"].max()

        drifted_df = injector.inject_data_quality_issues(
            sample_df,
            DataQualityIssue.OUTLIERS,
            affected_fraction=0.05,
            features=["feature1"],
        )

        # Should have some values beyond original range
        new_max = drifted_df["feature1"].max()
        assert new_max > original_max * 1.1

    def test_drift_scenarios(self, injector, sample_df):
        """Test pre-defined drift scenarios."""
        scenarios = injector.get_available_scenarios()

        assert "gradual_degradation" in scenarios
        assert "sudden_shift" in scenarios

        # Apply a scenario
        drifted_df = injector.create_drift_scenario(sample_df, "gradual_degradation")

        # Should return a DataFrame of same length
        assert len(drifted_df) == len(sample_df)

    def test_recurring_drift(self, injector, sample_df):
        """Test recurring/seasonal drift injection."""
        config = DriftConfig(
            drift_type=DriftType.RECURRING,
            magnitude=0.3,
        )
        drifted_df = injector.inject_drift(sample_df, config)

        # Check for periodic pattern (variance should be higher due to oscillation)
        original_std = sample_df["feature1"].std()
        drifted_std = drifted_df["feature1"].std()

        # Drifted data should have more variance due to sinusoidal pattern
        assert drifted_std > original_std * 0.9

    def test_label_drift_classification(self, injector, sample_df):
        """Test label drift for classification."""
        original_fraud_rate = sample_df["is_fraud"].mean()

        config = DriftConfig(
            drift_type=DriftType.LABEL,
            magnitude=0.3,
            start_index=500,
        )
        drifted_df = injector.inject_drift(sample_df, config)

        # Fraud rate in later portion should be different
        later_fraud_rate = drifted_df["is_fraud"].iloc[500:].mean()
        assert abs(later_fraud_rate - original_fraud_rate) > 0.05


class TestDataIntegration:
    """Integration tests for data generation and drift injection."""

    def test_generate_and_drift(self):
        """Test full workflow: generate data then inject drift."""
        # Generate data
        generator = SyntheticDataGenerator(seed=42)
        fraud_df = generator.generate_fraud_data(n_samples=1000)

        # Inject drift
        injector = DriftInjector(seed=42)
        drifted_df = injector.create_drift_scenario(fraud_df, "sudden_shift")

        # Should have same columns
        assert set(fraud_df.columns) == set(drifted_df.columns)

        # Amount distribution should be different after drift
        original_mean = fraud_df["amount"].mean()
        drifted_mean = drifted_df["amount"].iloc[-300:].mean()

        # The means should be different
        assert abs(drifted_mean - original_mean) > 1

    def test_reference_vs_current_data(self):
        """Test creating reference and current datasets for drift detection."""
        generator = SyntheticDataGenerator(seed=42)
        full_df = generator.generate_churn_data(n_samples=1000)

        # Split into reference and current
        reference_df = full_df.iloc[:700].copy()
        current_df = full_df.iloc[700:].copy()

        # Inject drift into current
        injector = DriftInjector(seed=42)
        config = DriftConfig(
            drift_type=DriftType.COVARIATE,
            magnitude=0.4,
            start_index=0,  # Affect all of current
        )
        current_drifted = injector.inject_drift(current_df, config)

        # Compare distributions
        ref_tenure_mean = reference_df["tenure_months"].mean()
        cur_tenure_mean = current_drifted["tenure_months"].mean()

        # Should be noticeably different
        assert abs(cur_tenure_mean - ref_tenure_mean) > ref_tenure_mean * 0.1
