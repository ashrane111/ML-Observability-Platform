"""
Data Module

Provides synthetic data generation and drift injection capabilities.
"""

from src.data.drift_injector import (
    DataQualityIssue,
    DriftConfig,
    DriftInjector,
    DriftType,
    inject_gradual_drift,
    inject_missing_values,
    inject_sudden_drift,
)
from src.data.synthetic_generator import SyntheticDataGenerator, generate_datasets

__all__ = [
    # Generator
    "SyntheticDataGenerator",
    "generate_datasets",
    # Drift Injector
    "DriftInjector",
    "DriftConfig",
    "DriftType",
    "DataQualityIssue",
    # Convenience functions
    "inject_gradual_drift",
    "inject_sudden_drift",
    "inject_missing_values",
]
