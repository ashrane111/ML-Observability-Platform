#!/usr/bin/env python
"""
ML Observability Platform - Interactive Demo Script

This script demonstrates the full capabilities of the platform.

Usage:
    python scripts/demo.py [--skip-training] [--api-url URL]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.data.drift_injector import DriftConfig, DriftInjector, DriftType  # noqa: E402
from src.data.synthetic_generator import SyntheticDataGenerator  # noqa: E402
from src.models.churn_predictor import ChurnPredictor  # noqa: E402
from src.models.fraud_detector import FraudDetector  # noqa: E402
from src.models.preprocessing import (  # noqa: E402
    FeaturePreprocessor,
    prepare_churn_features,
    prepare_fraud_features,
    prepare_price_features,
)
from src.models.price_predictor import PricePredictor  # noqa: E402
from src.monitoring.alerts import AlertManager, AlertSeverity, AlertType  # noqa: E402, F401
from src.monitoring.drift_detector import DriftDetector  # noqa: E402


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n--- {title} ---\n")


def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    """Print metrics in a formatted table."""
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")
    print("-" * 40)


def demo_data_generation() -> dict[str, pd.DataFrame]:
    """Demonstrate data generation capabilities."""
    print_header("PHASE 1: Data Generation")

    generator = SyntheticDataGenerator(seed=42)
    datasets = {}

    # Generate fraud data
    print_subheader("Generating Fraud Detection Data")
    fraud_data = generator.generate_fraud_data(n_samples=2000)
    datasets["fraud"] = fraud_data
    print(f"Generated {len(fraud_data)} fraud detection samples")
    fraud_rate = fraud_data["is_fraud"].mean()
    print(f"  Fraud rate: {fraud_rate:.2%}")

    # Generate price data
    print_subheader("Generating Price Prediction Data")
    price_data = generator.generate_price_data(n_samples=1000)
    datasets["price"] = price_data
    print(f"Generated {len(price_data)} price prediction samples")
    print(f"  Price range: ${price_data['price'].min():,.0f} - ${price_data['price'].max():,.0f}")

    # Generate churn data
    print_subheader("Generating Churn Prediction Data")
    churn_data = generator.generate_churn_data(n_samples=1500)
    datasets["churn"] = churn_data
    print(f"Generated {len(churn_data)} churn prediction samples")
    churn_rate = churn_data["churned"].mean()
    print(f"  Churn rate: {churn_rate:.2%}")

    return datasets


def demo_model_training(datasets: dict[str, pd.DataFrame]) -> dict:
    """Demonstrate model training."""
    print_header("PHASE 2: Model Training")

    models: dict[str, Any] = {}
    preprocessors: dict[str, FeaturePreprocessor] = {}

    # Train Fraud Detector
    print_subheader("Training Fraud Detector (XGBoost)")
    fraud_data = datasets["fraud"]
    X_fraud_raw, y_fraud = prepare_fraud_features(fraud_data)

    fraud_preprocessor = FeaturePreprocessor()
    X_fraud = fraud_preprocessor.fit_transform(X_fraud_raw)

    split_idx = int(len(X_fraud) * 0.8)
    X_train_fraud, X_test_fraud = X_fraud[:split_idx], X_fraud[split_idx:]
    y_train_fraud, y_test_fraud = y_fraud[:split_idx], y_fraud[split_idx:]

    fraud_model = FraudDetector()
    fraud_model.fit(X_train_fraud, y_train_fraud)
    fraud_metrics = fraud_model.evaluate(X_test_fraud, y_test_fraud)
    models["fraud"] = fraud_model
    preprocessors["fraud"] = fraud_preprocessor
    print_metrics(fraud_metrics, "Fraud Detector Performance")

    # Train Price Predictor
    print_subheader("Training Price Predictor (LightGBM)")
    price_data = datasets["price"]
    X_price_raw, y_price = prepare_price_features(price_data)

    price_preprocessor = FeaturePreprocessor()
    X_price = price_preprocessor.fit_transform(X_price_raw)

    split_idx = int(len(X_price) * 0.8)
    X_train_price, X_test_price = X_price[:split_idx], X_price[split_idx:]
    y_train_price, y_test_price = y_price[:split_idx], y_price[split_idx:]

    price_model = PricePredictor()
    price_model.fit(X_train_price, y_train_price)
    price_metrics = price_model.evaluate(X_test_price, y_test_price)
    models["price"] = price_model
    preprocessors["price"] = price_preprocessor
    print_metrics(price_metrics, "Price Predictor Performance")

    # Train Churn Predictor
    print_subheader("Training Churn Predictor (Random Forest)")
    churn_data = datasets["churn"]
    X_churn_raw, y_churn = prepare_churn_features(churn_data)

    churn_preprocessor = FeaturePreprocessor()
    X_churn = churn_preprocessor.fit_transform(X_churn_raw)

    split_idx = int(len(X_churn) * 0.8)
    X_train_churn, X_test_churn = X_churn[:split_idx], X_churn[split_idx:]
    y_train_churn, y_test_churn = y_churn[:split_idx], y_churn[split_idx:]

    churn_model = ChurnPredictor()
    churn_model.fit(X_train_churn, y_train_churn)
    churn_metrics = churn_model.evaluate(X_test_churn, y_test_churn)
    models["churn"] = churn_model
    preprocessors["churn"] = churn_preprocessor
    print_metrics(churn_metrics, "Churn Predictor Performance")

    return {"models": models, "preprocessors": preprocessors}


def demo_drift_detection(datasets: dict[str, pd.DataFrame]) -> None:
    """Demonstrate drift detection capabilities."""
    print_header("PHASE 3: Drift Detection")

    generator = SyntheticDataGenerator(seed=123)
    injector = DriftInjector(seed=123)

    # Setup drift detector with fraud data as reference
    print_subheader("Setting Up Drift Detector")
    reference_data = datasets["fraud"]
    detector = DriftDetector(
        psi_threshold_warning=0.1,
        psi_threshold_critical=0.2,
        drift_share_threshold=0.3,
    )
    detector.set_reference_data(reference_data, target_column="is_fraud")
    print(f"Reference data set: {len(reference_data)} samples")

    # Test 1: No drift scenario
    print_subheader("Scenario 1: No Drift (Similar Distribution)")
    current_no_drift = generator.generate_fraud_data(n_samples=500)
    result_no_drift = detector.detect_drift(current_no_drift, dataset_name="no_drift")
    print(f"  Drift Status: {result_no_drift.drift_status.value}")
    print(f"  Drift Share: {result_no_drift.drift_share:.2%}")
    print(f"  Drifted Features: {len(result_no_drift.drifted_features)}")

    # Test 2: Gradual drift scenario
    print_subheader("Scenario 2: Gradual Drift")
    current_gradual = generator.generate_fraud_data(n_samples=500)
    drift_config = DriftConfig(
        drift_type=DriftType.GRADUAL,
        features=["amount", "distance_from_home"],
        magnitude=0.5,
        affected_fraction=0.5,
    )
    current_gradual = injector.inject_drift(current_gradual, drift_config)
    result_gradual = detector.detect_drift(current_gradual, dataset_name="gradual_drift")
    print(f"  Drift Status: {result_gradual.drift_status.value}")
    print(f"  Drift Share: {result_gradual.drift_share:.2%}")
    print(f"  Drifted Features: {result_gradual.drifted_features}")

    # Test 3: Sudden drift scenario
    print_subheader("Scenario 3: Sudden Drift (Major Distribution Shift)")
    current_sudden = generator.generate_fraud_data(n_samples=500)
    drift_config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        features=["amount", "distance_from_home", "hour_of_day"],
        magnitude=0.8,
        affected_fraction=0.8,
    )
    current_sudden = injector.inject_drift(current_sudden, drift_config)
    result_sudden = detector.detect_drift(current_sudden, dataset_name="sudden_drift")
    print(f"  Drift Status: {result_sudden.drift_status.value}")
    print(f"  Drift Share: {result_sudden.drift_share:.2%}")
    print(f"  Drifted Features: {result_sudden.drifted_features}")


def demo_alerting() -> None:
    """Demonstrate alerting capabilities."""
    print_header("PHASE 4: Alert Management")

    alert_manager = AlertManager()

    print_subheader("Creating Alerts")

    # Create drift alert
    drift_alert = alert_manager.create_drift_alert(
        model_name="fraud_detector",
        drift_share=0.35,
        drifted_features=["amount", "distance_from_home", "hour_of_day"],
        dataset_name="production",
    )
    if drift_alert:
        print(f"Created drift alert: {drift_alert.title}")
        print(f"  Severity: {drift_alert.severity.value}")
        alert_id_short = drift_alert.alert_id[:8]
        print(f"  Alert ID: {alert_id_short}...")

    # Get alert summary
    print_subheader("Alert Summary")
    summary = alert_manager.get_alert_summary()
    print(f"  Total Active Alerts: {summary['total_active']}")
    print(f"  By Severity: {summary['by_severity']}")
    print(f"  By Type: {summary['by_type']}")

    # Demonstrate alert lifecycle
    if drift_alert:
        print_subheader("Alert Lifecycle Demo")

        # Acknowledge
        alert_manager.acknowledge_alert(drift_alert.alert_id, acknowledged_by="demo_user")
        print("Alert acknowledged by demo_user")

        # Resolve
        alert_manager.resolve_alert(
            drift_alert.alert_id,
            resolved_by="demo_user",
            resolution_notes="Retrained model with new data distribution",
        )
        print("Alert resolved with notes")


def demo_predictions(trained_components: dict) -> None:
    """Demonstrate making predictions."""
    print_header("PHASE 5: Making Predictions")

    models = trained_components["models"]
    preprocessors = trained_components["preprocessors"]
    generator = SyntheticDataGenerator(seed=999)

    # Fraud predictions
    print_subheader("Fraud Detection Predictions")
    fraud_model = models["fraud"]
    fraud_preprocessor = preprocessors["fraud"]

    # Create sample transactions
    sample_transactions = generator.generate_fraud_data(n_samples=5)
    X_raw, _ = prepare_fraud_features(sample_transactions)
    X_sample = fraud_preprocessor.transform(X_raw)

    probabilities = fraud_model.predict_fraud_probability(X_sample)
    predictions = fraud_model.predict(X_sample)

    print("Sample Transaction Predictions:")
    print("-" * 50)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
        status = "FRAUD" if pred == 1 else "LEGIT"
        print(f"  Transaction {i+1}: {status} (Probability: {prob:.2%}, Risk: {risk})")

    # Price predictions
    print_subheader("Price Predictions")
    price_model = models["price"]
    price_preprocessor = preprocessors["price"]

    sample_properties = generator.generate_price_data(n_samples=3)
    X_raw, _ = prepare_price_features(sample_properties)
    X_price = price_preprocessor.transform(X_raw)

    price_predictions = price_model.predict(X_price)
    actual_prices = sample_properties["price"].values

    print("Sample Property Predictions:")
    print("-" * 50)
    for i, (pred, actual) in enumerate(zip(price_predictions, actual_prices)):
        error_pct = abs(pred - actual) / actual * 100
        print(
            f"  Property {i+1}: Predicted ${pred:,.0f} | Actual ${actual:,.0f} | Error: {error_pct:.1f}%"  # noqa: E501
        )

    # Churn predictions
    print_subheader("Churn Predictions")
    churn_model = models["churn"]
    churn_preprocessor = preprocessors["churn"]

    sample_customers = generator.generate_churn_data(n_samples=5)
    X_raw, _ = prepare_churn_features(sample_customers)
    X_churn = churn_preprocessor.transform(X_raw)

    churn_probs = churn_model.predict_churn_probability(X_churn)
    churn_preds = churn_model.predict(X_churn)

    print("Sample Customer Predictions:")
    print("-" * 50)
    for i, (pred, prob) in enumerate(zip(churn_preds, churn_probs)):
        segment = "CRITICAL" if prob > 0.8 else "AT RISK" if prob > 0.5 else "STABLE"
        status = "WILL CHURN" if pred == 1 else "WILL STAY"
        print(f"  Customer {i+1}: {status} (Probability: {prob:.2%}, Segment: {segment})")


def demo_api_interaction(api_url: str) -> None:
    """Demonstrate API interactions."""
    print_header("PHASE 6: API Interaction")

    try:
        import requests
    except ImportError:
        print("requests library not installed. Skipping API demo.")
        return

    print(f"Connecting to API at {api_url}...")

    # Health check
    print_subheader("Health Check")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("API is healthy")
            print(f"  Status: {health['status']}")
            print(f"  Version: {health['version']}")
            print(f"  Models Loaded: {health['models_loaded']}")
        else:
            print(f"Health check returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Is it running?")
        print("  Start with: make run-api")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    # List models
    print_subheader("Available Models")
    try:
        response = requests.get(f"{api_url}/predict/models", timeout=5)
        if response.status_code == 200:
            models = response.json()["models"]
            for name, info in models.items():
                status = "Loaded" if info["loaded"] else "Not loaded"
                print(f"  {name}: {status}")
    except Exception as e:
        print(f"Error: {e}")


def main() -> int:
    """Run the interactive demo."""
    parser = argparse.ArgumentParser(description="ML Observability Platform Demo")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training phase",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL for interaction demo",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API interaction demo",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" ML OBSERVABILITY PLATFORM - INTERACTIVE DEMO")
    print("=" * 70)
    print("\nThis demo will walk you through all platform capabilities.\n")

    start_time = time.time()

    try:
        # Phase 1: Data Generation
        datasets = demo_data_generation()

        # Phase 2: Model Training
        if not args.skip_training:
            trained_components = demo_model_training(datasets)
        else:
            print_header("PHASE 2: Model Training (SKIPPED)")
            trained_components = None

        # Phase 3: Drift Detection
        demo_drift_detection(datasets)

        # Phase 4: Alerting
        demo_alerting()

        # Phase 5: Predictions
        if trained_components:
            demo_predictions(trained_components)

        # Phase 6: API Interaction
        if not args.skip_api:
            demo_api_interaction(args.api_url)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1

    elapsed_time = time.time() - start_time

    print_header("DEMO COMPLETE")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print("\nNext steps:")
    print("  1. Start the API: make run-api-dev")
    print("  2. Start Docker stack: make docker-up")
    print("  3. View API docs: http://localhost:8000/docs")
    print("  4. View Grafana: http://localhost:3000 (admin/admin)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
