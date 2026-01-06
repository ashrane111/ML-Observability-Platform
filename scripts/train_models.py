#!/usr/bin/env python3
"""
Model Training Script

Trains all three ML models with MLflow tracking:
- Fraud Detector (XGBoost)
- Price Predictor (LightGBM)
- Churn Predictor (Random Forest)

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --model fraud
    python scripts/train_models.py --model all --experiment my_experiment
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.models.churn_predictor import ChurnPredictor  # noqa: E402
from src.models.fraud_detector import FraudDetector  # noqa: E402
from src.models.preprocessing import (  # noqa: E402
    FeaturePreprocessor,
    prepare_churn_features,
    prepare_fraud_features,
    prepare_price_features,
)
from src.models.price_predictor import PricePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML models with MLflow tracking")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "fraud", "price", "churn"],
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing training data (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ml-observability-training",
        help="MLflow experiment name (default: ml-observability-training)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI (default: mlruns - local directory)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    return parser.parse_args()


def setup_mlflow(experiment_name: str, tracking_uri: str) -> None:
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info(f"MLflow tracking URI: {tracking_uri}")


def load_data(data_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load dataset from parquet or csv."""
    parquet_path = data_dir / f"{dataset_name}_data.parquet"
    csv_path = data_dir / f"{dataset_name}_data.csv"

    if parquet_path.exists():
        logger.info(f"Loading {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info(f"Loading {csv_path}")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No data file found for {dataset_name} in {data_dir}. "
            "Run 'python scripts/generate_data.py' first."
        )


def train_fraud_detector(
    data_dir: Path,
    output_dir: Path,
    test_size: float,
    seed: int,
    use_mlflow: bool,
) -> dict:
    """Train the fraud detection model."""
    logger.info("=" * 60)
    logger.info("Training Fraud Detector")
    logger.info("=" * 60)

    # Load data
    df = load_data(data_dir, "fraud")
    X, y = prepare_fraud_features(df)

    # Preprocess features
    preprocessor = FeaturePreprocessor(
        numerical_strategy="standard",
        categorical_strategy="onehot",
    )
    X_processed = preprocessor.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Fraud rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # Create and train model
    model = FraudDetector(version="1.0.0")

    if use_mlflow:
        with mlflow.start_run(run_name="fraud_detector"):
            # Log parameters
            mlflow.log_params(model.hyperparameters)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("fraud_rate_train", y_train.mean())

            # Train
            model.fit(X_train, y_train, validation_data=(X_test, y_test))

            # Evaluate
            metrics = model.evaluate(X_test, y_test, prefix="test")
            mlflow.log_metrics(metrics)

            # Log feature importance
            importance = model.get_feature_importance()
            if importance:
                for feat, imp in importance.items():
                    mlflow.log_metric(f"importance_{feat}", imp)

            # Save model
            model_path = model.save(output_dir / "fraud_detector")
            mlflow.log_artifact(str(model_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(model.model, "model")
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test))
        metrics = model.evaluate(X_test, y_test, prefix="test")
        model.save(output_dir / "fraud_detector")

    logger.info(f"Fraud Detector metrics: {metrics}")
    return metrics


def train_price_predictor(
    data_dir: Path,
    output_dir: Path,
    test_size: float,
    seed: int,
    use_mlflow: bool,
) -> dict:
    """Train the price prediction model."""
    logger.info("=" * 60)
    logger.info("Training Price Predictor")
    logger.info("=" * 60)

    # Load data
    df = load_data(data_dir, "price")
    X, y = prepare_price_features(df)

    # Preprocess features
    preprocessor = FeaturePreprocessor(
        numerical_strategy="standard",
        categorical_strategy="onehot",
    )
    X_processed = preprocessor.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=seed
    )

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")

    # Create and train model
    model = PricePredictor(version="1.0.0")

    if use_mlflow:
        with mlflow.start_run(run_name="price_predictor"):
            # Log parameters
            mlflow.log_params(model.hyperparameters)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("price_mean", y.mean())
            mlflow.log_param("price_std", y.std())

            # Train
            model.fit(X_train, y_train, validation_data=(X_test, y_test))

            # Evaluate
            metrics = model.evaluate(X_test, y_test, prefix="test")
            mlflow.log_metrics(metrics)

            # Log feature importance
            importance = model.get_feature_importance()
            if importance:
                for feat, imp in importance.items():
                    mlflow.log_metric(f"importance_{feat}", imp)

            # Save model
            model_path = model.save(output_dir / "price_predictor")
            mlflow.log_artifact(str(model_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(model.model, "model")
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test))
        metrics = model.evaluate(X_test, y_test, prefix="test")
        model.save(output_dir / "price_predictor")

    logger.info(f"Price Predictor metrics: {metrics}")
    return metrics


def train_churn_predictor(
    data_dir: Path,
    output_dir: Path,
    test_size: float,
    seed: int,
    use_mlflow: bool,
) -> dict:
    """Train the churn prediction model."""
    logger.info("=" * 60)
    logger.info("Training Churn Predictor")
    logger.info("=" * 60)

    # Load data
    df = load_data(data_dir, "churn")
    X, y = prepare_churn_features(df)

    # Preprocess features
    preprocessor = FeaturePreprocessor(
        numerical_strategy="standard",
        categorical_strategy="onehot",
    )
    X_processed = preprocessor.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # Create and train model
    model = ChurnPredictor(version="1.0.0")

    if use_mlflow:
        with mlflow.start_run(run_name="churn_predictor"):
            # Log parameters
            mlflow.log_params(model.hyperparameters)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("churn_rate_train", y_train.mean())

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            metrics = model.evaluate(X_test, y_test, prefix="test")
            mlflow.log_metrics(metrics)

            # Log feature importance
            importance = model.get_feature_importance()
            if importance:
                for feat, imp in importance.items():
                    mlflow.log_metric(f"importance_{feat}", imp)

            # Save model
            model_path = model.save(output_dir / "churn_predictor")
            mlflow.log_artifact(str(model_path))

            # Log model to MLflow
            mlflow.sklearn.log_model(model.model, "model")
    else:
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test, prefix="test")
        model.save(output_dir / "churn_predictor")

    logger.info(f"Churn Predictor metrics: {metrics}")
    return metrics


def main() -> None:
    """Main training function."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ML Observability Platform - Model Training")
    logger.info("=" * 60)

    # Setup MLflow
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        setup_mlflow(args.experiment, args.mlflow_tracking_uri)

    all_metrics = {}

    # Train requested models
    if args.model in ["all", "fraud"]:
        all_metrics["fraud"] = train_fraud_detector(
            data_dir, output_dir, args.test_size, args.seed, use_mlflow
        )

    if args.model in ["all", "price"]:
        all_metrics["price"] = train_price_predictor(
            data_dir, output_dir, args.test_size, args.seed, use_mlflow
        )

    if args.model in ["all", "churn"]:
        all_metrics["churn"] = train_churn_predictor(
            data_dir, output_dir, args.test_size, args.seed, use_mlflow
        )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    for model_name, metrics in all_metrics.items():
        logger.info(f"\n{model_name.upper()} Model:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {value}")

    logger.info(f"\nModels saved to: {output_dir}")
    if use_mlflow:
        logger.info(f"MLflow runs saved to: {args.mlflow_tracking_uri}")
        logger.info("View with: mlflow ui")


if __name__ == "__main__":
    main()
