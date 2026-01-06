"""
Drift Injector Module

Injects various types of drift into datasets for testing and demonstration.
Supports:
- Gradual drift (slow distribution shift over time)
- Sudden drift (abrupt distribution change)
- Feature drift (individual feature corruption)
- Label drift (target distribution shift)
- Concept drift (feature-target relationship change)
- Data quality issues (missing values, outliers)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class DriftType(str, Enum):
    """Types of drift that can be injected."""

    GRADUAL = "gradual"  # Slow distribution shift
    SUDDEN = "sudden"  # Abrupt distribution change
    INCREMENTAL = "incremental"  # Small continuous changes
    RECURRING = "recurring"  # Periodic/seasonal drift
    FEATURE = "feature"  # Individual feature drift
    LABEL = "label"  # Target distribution shift
    CONCEPT = "concept"  # Feature-target relationship change
    COVARIATE = "covariate"  # Input distribution shift


class DataQualityIssue(str, Enum):
    """Types of data quality issues."""

    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    SCHEMA_VIOLATION = "schema_violation"
    NOISE = "noise"


@dataclass
class DriftConfig:
    """Configuration for drift injection."""

    drift_type: DriftType
    magnitude: float = 0.3  # Strength of drift (0-1)
    affected_fraction: float = 0.5  # Fraction of data affected
    features: Optional[list[str]] = None  # Specific features to affect
    start_index: Optional[int] = None  # Where drift starts
    transition_length: Optional[int] = None  # For gradual drift
    seed: int = 42


@dataclass
class DriftReport:
    """Report of injected drift."""

    drift_type: DriftType
    magnitude: float
    affected_features: list[str]
    affected_samples: int
    original_stats: dict
    drifted_stats: dict


class DriftInjector:
    """
    Injects controlled drift into datasets for ML monitoring testing.

    Supports multiple drift types to simulate real-world scenarios:
    - Production data distribution changes
    - Seasonal variations
    - Upstream data issues
    - Model degradation scenarios
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the drift injector.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._drift_history: list[DriftReport] = []
        logger.info(f"DriftInjector initialized with seed={seed}")

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random seed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    # =========================================================================
    # Main Drift Injection Methods
    # =========================================================================

    def inject_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject drift into a DataFrame based on configuration.

        Args:
            df: Input DataFrame
            config: Drift configuration

        Returns:
            DataFrame with drift injected
        """
        df = df.copy()

        drift_methods = {
            DriftType.GRADUAL: self._inject_gradual_drift,
            DriftType.SUDDEN: self._inject_sudden_drift,
            DriftType.INCREMENTAL: self._inject_incremental_drift,
            DriftType.RECURRING: self._inject_recurring_drift,
            DriftType.FEATURE: self._inject_feature_drift,
            DriftType.LABEL: self._inject_label_drift,
            DriftType.CONCEPT: self._inject_concept_drift,
            DriftType.COVARIATE: self._inject_covariate_drift,
        }

        method = drift_methods.get(config.drift_type)
        if method is None:
            raise ValueError(f"Unknown drift type: {config.drift_type}")

        logger.info(f"Injecting {config.drift_type.value} drift with magnitude={config.magnitude}")
        return method(df, config)

    def inject_data_quality_issues(
        self,
        df: pd.DataFrame,
        issue_type: DataQualityIssue,
        affected_fraction: float = 0.1,
        features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Inject data quality issues into a DataFrame.

        Args:
            df: Input DataFrame
            issue_type: Type of data quality issue
            affected_fraction: Fraction of data to affect
            features: Specific features to affect (None = all numeric)

        Returns:
            DataFrame with data quality issues
        """
        df = df.copy()

        issue_methods = {
            DataQualityIssue.MISSING_VALUES: self._inject_missing_values,
            DataQualityIssue.OUTLIERS: self._inject_outliers,
            DataQualityIssue.DUPLICATES: self._inject_duplicates,
            DataQualityIssue.NOISE: self._inject_noise,
        }

        method = issue_methods.get(issue_type)
        if method is None:
            raise ValueError(f"Unknown data quality issue: {issue_type}")

        logger.info(f"Injecting {issue_type.value} affecting {affected_fraction*100:.1f}% of data")
        return method(df, affected_fraction, features)

    # =========================================================================
    # Drift Type Implementations
    # =========================================================================

    def _inject_gradual_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject gradual drift that increases over time.

        The distribution slowly shifts from the original to a drifted state.
        """
        features = config.features or self._get_numeric_features(df)
        n = len(df)

        # Default: drift starts at 20% of data
        start_idx = config.start_index or int(n * 0.2)
        transition_len = config.transition_length or int(n * 0.6)

        # Create drift weights that gradually increase
        weights = np.zeros(n)
        end_idx = min(start_idx + transition_len, n)
        weights[start_idx:end_idx] = np.linspace(0, 1, end_idx - start_idx)
        weights[end_idx:] = 1.0

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            # original_mean = df[feature].mean()
            original_std = df[feature].std()

            # Shift based on magnitude and weight
            shift = original_std * config.magnitude * 2
            df[feature] = df[feature] + shift * weights

        logger.debug(f"Gradual drift injected into {len(features)} features")
        return df

    def _inject_sudden_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject sudden drift that changes abruptly at a point.

        The distribution changes suddenly, simulating a deployment issue
        or upstream data change.
        """
        features = config.features or self._get_numeric_features(df)
        n = len(df)

        # Default: drift starts at 50% of data
        start_idx = config.start_index or int(n * 0.5)

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            original_std = df[feature].std()

            # Apply sudden shift
            shift = original_std * config.magnitude * 3
            df.loc[start_idx:, feature] = df.loc[start_idx:, feature] + shift

        logger.debug(f"Sudden drift injected at index {start_idx}")
        return df

    def _inject_incremental_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject small continuous incremental changes.

        Each data point is slightly different from the previous,
        simulating slow environmental changes.
        """
        features = config.features or self._get_numeric_features(df)
        n = len(df)

        # Create cumulative drift
        cumulative_factor = np.cumsum(np.ones(n)) / n * config.magnitude

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            original_std = df[feature].std()
            df[feature] = df[feature] * (1 + cumulative_factor * original_std / df[feature].mean())

        logger.debug(f"Incremental drift injected into {len(features)} features")
        return df

    def _inject_recurring_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject periodic/seasonal drift patterns.

        Simulates seasonal variations in data.
        """
        features = config.features or self._get_numeric_features(df)
        n = len(df)

        # Create sinusoidal pattern (simulating seasonality)
        # period = n // 4  # 4 complete cycles
        pattern = np.sin(np.linspace(0, 8 * np.pi, n)) * config.magnitude

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            original_std = df[feature].std()
            df[feature] = df[feature] + pattern * original_std

        logger.debug("Recurring drift injected with sinusoidal pattern")
        return df

    def _inject_feature_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject drift into specific features only.

        Simulates issues with specific data sources or sensors.
        """
        if not config.features:
            # Select random features
            numeric_features = self._get_numeric_features(df)
            n_features = max(1, int(len(numeric_features) * 0.3))
            features = list(self.rng.choice(numeric_features, size=n_features, replace=False))
        else:
            features = config.features

        n = len(df)
        affected_mask = self.rng.random(n) < config.affected_fraction

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            # original_mean = df[feature].mean()
            original_std = df[feature].std()

            # Different drift patterns for different features
            drift_pattern = self.rng.choice(["shift", "scale", "noise"])

            if drift_pattern == "shift":
                shift = original_std * config.magnitude * 2
                df.loc[affected_mask, feature] += shift
            elif drift_pattern == "scale":
                scale = 1 + config.magnitude
                df.loc[affected_mask, feature] *= scale
            else:  # noise
                noise = self.rng.normal(0, original_std * config.magnitude, affected_mask.sum())
                df.loc[affected_mask, feature] += noise

        logger.debug(f"Feature drift injected into: {features}")
        return df

    def _inject_label_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject drift in target variable distribution.

        Simulates changes in the outcome distribution (e.g., more fraud,
        higher prices, more churn).
        """
        # Try to identify target column
        target_candidates = ["is_fraud", "churned", "price", "target", "label"]
        target_col = None

        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            logger.warning("No target column found for label drift")
            return df

        n = len(df)
        start_idx = config.start_index or int(n * 0.5)

        if df[target_col].dtype == bool or df[target_col].nunique() == 2:
            # Binary classification - flip some labels
            flip_mask = (df.index >= start_idx) & (self.rng.random(n) < config.magnitude)
            df.loc[flip_mask, target_col] = ~df.loc[flip_mask, target_col]
            logger.debug(f"Flipped {flip_mask.sum()} labels")
        else:
            # Regression - shift values
            original_std = df[target_col].std()
            shift = original_std * config.magnitude * 2
            df.loc[start_idx:, target_col] += shift

        return df

    def _inject_concept_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject concept drift - change the relationship between features and target.

        This is the most insidious type of drift where the patterns the model
        learned no longer hold.
        """
        target_candidates = ["is_fraud", "churned", "price", "target", "label"]
        target_col = None

        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            logger.warning("No target column found for concept drift")
            return df

        features = config.features or self._get_numeric_features(df)
        features = [f for f in features if f != target_col]

        n = len(df)
        start_idx = config.start_index or int(n * 0.5)

        # Change feature values based on target in the drifted portion
        # This inverts or changes the feature-target relationship
        for feature in features[:3]:  # Affect top 3 features
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            # Swap feature values between positive and negative classes
            drifted_portion = df.index >= start_idx

            if df[target_col].dtype == bool:
                positive_mask = drifted_portion & df[target_col]
                negative_mask = drifted_portion & ~df[target_col]

                # Add offset to positive class, subtract from negative
                feature_std = df[feature].std()
                df.loc[positive_mask, feature] -= feature_std * config.magnitude
                df.loc[negative_mask, feature] += feature_std * config.magnitude

        logger.debug("Concept drift injected, changing feature-target relationships")
        return df

    def _inject_covariate_drift(
        self,
        df: pd.DataFrame,
        config: DriftConfig,
    ) -> pd.DataFrame:
        """
        Inject covariate drift - change in input feature distributions.

        The feature distributions change but the relationship with target
        remains the same.
        """
        features = config.features or self._get_numeric_features(df)
        n = len(df)
        start_idx = config.start_index or int(n * 0.5)

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            # Get stats from the pre-drift portion
            pre_drift_data = df.loc[:start_idx, feature]
            original_mean = pre_drift_data.mean()
            original_std = pre_drift_data.std()

            # Skip if no variance or invalid stats
            if original_std == 0 or np.isnan(original_std) or np.isnan(original_mean):
                continue

            # Shift the distribution
            shift = original_std * config.magnitude * 2
            df.loc[start_idx:, feature] = df.loc[start_idx:, feature] + shift

        logger.debug(f"Covariate drift injected into {len(features)} features")
        return df

    # =========================================================================
    # Data Quality Issue Implementations
    # =========================================================================

    def _inject_missing_values(
        self,
        df: pd.DataFrame,
        affected_fraction: float,
        features: Optional[list[str]],
    ) -> pd.DataFrame:
        """Inject missing values (NaN) into data."""
        features = features or self._get_numeric_features(df)
        n = len(df)

        for feature in features:
            if feature not in df.columns:
                continue

            missing_mask = self.rng.random(n) < affected_fraction
            df.loc[missing_mask, feature] = np.nan

        logger.debug(f"Injected missing values into {len(features)} features")
        return df

    def _inject_outliers(
        self,
        df: pd.DataFrame,
        affected_fraction: float,
        features: Optional[list[str]],
    ) -> pd.DataFrame:
        """Inject outliers (extreme values) into data."""
        features = features or self._get_numeric_features(df)
        n = len(df)

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            outlier_mask = self.rng.random(n) < affected_fraction
            n_outliers = outlier_mask.sum()

            if n_outliers == 0:
                continue

            # Generate outliers at 3-5 standard deviations
            mean = df[feature].mean()
            std = df[feature].std()

            outlier_values = mean + self.rng.choice([-1, 1], n_outliers) * std * self.rng.uniform(
                3, 5, n_outliers
            )
            df.loc[outlier_mask, feature] = outlier_values

        logger.debug(f"Injected outliers into {len(features)} features")
        return df

    def _inject_duplicates(
        self,
        df: pd.DataFrame,
        affected_fraction: float,
        features: Optional[list[str]],
    ) -> pd.DataFrame:
        """Inject duplicate rows into data."""
        n_duplicates = int(len(df) * affected_fraction)

        if n_duplicates == 0:
            return df

        # Sample random rows to duplicate
        duplicate_indices = self.rng.choice(len(df), size=n_duplicates, replace=True)
        duplicates = df.iloc[duplicate_indices].copy()

        df = pd.concat([df, duplicates], ignore_index=True)
        logger.debug(f"Injected {n_duplicates} duplicate rows")
        return df

    def _inject_noise(
        self,
        df: pd.DataFrame,
        affected_fraction: float,
        features: Optional[list[str]],
    ) -> pd.DataFrame:
        """Inject random noise into data."""
        features = features or self._get_numeric_features(df)
        n = len(df)

        noise_mask = self.rng.random(n) < affected_fraction

        for feature in features:
            if feature not in df.columns:
                continue
            if not np.issubdtype(df[feature].dtype, np.number):
                continue

            std = df[feature].std()
            noise = self.rng.normal(0, std * 0.5, noise_mask.sum())
            df.loc[noise_mask, feature] += noise

        logger.debug(f"Injected noise into {len(features)} features")
        return df

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_numeric_features(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric feature columns."""
        exclude = ["id", "timestamp", "date", "created", "updated"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if not any(e in c.lower() for e in exclude)]

    def create_drift_scenario(
        self,
        df: pd.DataFrame,
        scenario: str = "gradual_degradation",
    ) -> pd.DataFrame:
        """
        Apply a pre-defined drift scenario.

        Args:
            df: Input DataFrame
            scenario: Name of the scenario

        Returns:
            DataFrame with drift applied
        """
        scenarios = {
            "gradual_degradation": [
                DriftConfig(DriftType.GRADUAL, magnitude=0.3),
            ],
            "sudden_shift": [
                DriftConfig(DriftType.SUDDEN, magnitude=0.5),
            ],
            "feature_corruption": [
                DriftConfig(DriftType.FEATURE, magnitude=0.4, affected_fraction=0.3),
            ],
            "concept_change": [
                DriftConfig(DriftType.CONCEPT, magnitude=0.4),
            ],
            "data_quality_degradation": [
                DriftConfig(DriftType.GRADUAL, magnitude=0.2),
            ],
            "seasonal_variation": [
                DriftConfig(DriftType.RECURRING, magnitude=0.3),
            ],
            "catastrophic_failure": [
                DriftConfig(DriftType.SUDDEN, magnitude=0.8),
                DriftConfig(DriftType.LABEL, magnitude=0.3),
            ],
        }

        if scenario not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")

        logger.info(f"Applying drift scenario: {scenario}")

        for config in scenarios[scenario]:
            df = self.inject_drift(df, config)

        return df

    def get_available_scenarios(self) -> list[str]:
        """Get list of available drift scenarios."""
        return [
            "gradual_degradation",
            "sudden_shift",
            "feature_corruption",
            "concept_change",
            "data_quality_degradation",
            "seasonal_variation",
            "catastrophic_failure",
        ]


# Convenience functions
def inject_gradual_drift(
    df: pd.DataFrame,
    magnitude: float = 0.3,
    features: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to inject gradual drift."""
    injector = DriftInjector(seed=seed)
    config = DriftConfig(
        drift_type=DriftType.GRADUAL,
        magnitude=magnitude,
        features=features,
    )
    return injector.inject_drift(df, config)


def inject_sudden_drift(
    df: pd.DataFrame,
    magnitude: float = 0.5,
    start_index: Optional[int] = None,
    features: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to inject sudden drift."""
    injector = DriftInjector(seed=seed)
    config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        magnitude=magnitude,
        start_index=start_index,
        features=features,
    )
    return injector.inject_drift(df, config)


def inject_missing_values(
    df: pd.DataFrame,
    fraction: float = 0.1,
    features: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to inject missing values."""
    injector = DriftInjector(seed=seed)
    return injector.inject_data_quality_issues(
        df, DataQualityIssue.MISSING_VALUES, fraction, features
    )
