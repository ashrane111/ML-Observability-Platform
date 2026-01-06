"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment for testing
os.environ["APP_ENV"] = "development"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture(scope="session")
def project_root_path():
    """Get the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root_path):
    """Get the data directory path."""
    return project_root_path / "data"


@pytest.fixture(scope="session")
def sample_fraud_data():
    """Generate sample fraud data for testing."""
    from src.data.synthetic_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    return generator.generate_fraud_data(n_samples=500, fraud_rate=0.05)


@pytest.fixture(scope="session")
def sample_price_data():
    """Generate sample price data for testing."""
    from src.data.synthetic_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    return generator.generate_price_data(n_samples=300)


@pytest.fixture(scope="session")
def sample_churn_data():
    """Generate sample churn data for testing."""
    from src.data.synthetic_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    return generator.generate_churn_data(n_samples=400, churn_rate=0.15)


@pytest.fixture(scope="session")
def all_sample_data(sample_fraud_data, sample_price_data, sample_churn_data):
    """Get all sample datasets."""
    return {
        "fraud": sample_fraud_data,
        "price": sample_price_data,
        "churn": sample_churn_data,
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "reference").mkdir()
    return data_dir
