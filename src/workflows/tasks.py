"""Prefect tasks for ML operations.

Reusable tasks for:
- Data loading and validation
- Model training and evaluation
- Drift detection
- Alerting and notifications
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from prefect import task
from prefect.artifacts import create_markdown_artifact, create_table_artifact  # noqa:F401

logger = logging.getLogger(__name__)


# ============================================================================
# Data Tasks
# ============================================================================


@task(
    name="load_data",
    description="Load data from file or database",
    retries=2,
    retry_delay_seconds=10,
    tags=["data", "io"],
)
def load_data(
    data_path: Union[str, Path],
    file_format: str = "parquet",
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load data from file.

    Args:
        data_path: Path to data file
        file_format: Format (parquet, csv, json)
        sample_size: Optional sample size for large datasets

    Returns:
        Loaded DataFrame
    """
    data_path = Path(data_path)
    logger.info(f"Loading data from {data_path}")

    if file_format == "parquet":
        df = pd.read_parquet(data_path)
    elif file_format == "csv":
        df = pd.read_csv(data_path)
    elif file_format == "json":
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows from {len(df)} total")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


@task(
    name="validate_data",
    description="Validate data quality",
    tags=["data", "validation"],
)
def validate_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    max_missing_pct: float = 0.1,
) -> Dict[str, Any]:
    """Validate data quality.

    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        max_missing_pct: Maximum allowed missing percentage per column

    Returns:
        Validation results dictionary
    """
    results = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "issues": [],
    }

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results["is_valid"] = False
            results["issues"].append(f"Missing required columns: {missing_cols}")

    # Check missing values
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct]
    if not high_missing.empty:
        results["issues"].append(f"Columns with high missing %: {high_missing.to_dict()}")

    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        results["issues"].append(f"Found {dup_count} duplicate rows")
        results["duplicate_count"] = dup_count

    results["missing_summary"] = missing_pct.to_dict()

    logger.info(
        f"Validation complete: valid={results['is_valid']}, issues={len(results['issues'])}"
    )
    return results


@task(
    name="prepare_features",
    description="Prepare features for model training",
    tags=["data", "preprocessing"],
)
def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for training.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        feature_columns: List of feature columns (None = all except target)

    Returns:
        Tuple of (X, y)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    y = df[target_column]

    if feature_columns:
        X = df[feature_columns]
    else:
        X = df.drop(columns=[target_column])

    logger.info(f"Prepared features: X shape={X.shape}, y shape={y.shape}")
    return X, y


# ============================================================================
# Model Tasks
# ============================================================================


@task(
    name="train_model",
    description="Train an ML model",
    retries=1,
    tags=["model", "training"],
)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train a model.

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model (fraud, price, churn)
        model_params: Optional model hyperparameters

    Returns:
        Trained model instance
    """
    # Import models - adjust path based on your project structure
    try:
        from src.models import ChurnPredictor, FraudDetector, PricePredictor
        from src.models.preprocessing import FeaturePreprocessor
    except ImportError:
        from models import ChurnPredictor, FraudDetector, PricePredictor
        from models.preprocessing import FeaturePreprocessor

    logger.info(f"Training {model_type} model with {len(X_train)} samples")

    # Initialize model
    model_classes = {
        "fraud": FraudDetector,
        "price": PricePredictor,
        "churn": ChurnPredictor,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_classes[model_type](**(model_params or {}))

    # Preprocess features
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.fit_transform(X_train)

    # Train
    model.fit(X_processed, y_train)

    logger.info(f"Model training complete")
    return model


@task(
    name="evaluate_model",
    description="Evaluate model performance",
    tags=["model", "evaluation"],
)
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> Dict[str, float]:
    """Evaluate model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name for logging

    Returns:
        Dictionary of metrics
    """
    try:
        from src.models.preprocessing import FeaturePreprocessor
    except ImportError:
        from models.preprocessing import FeaturePreprocessor

    logger.info(f"Evaluating {model_name} on {len(X_test)} samples")

    # Preprocess (use existing preprocessor if available)
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.fit_transform(X_test)

    # Get metrics from model
    metrics = model.evaluate(X_processed, y_test)

    # Create artifact
    create_markdown_artifact(
        key=f"{model_name}-evaluation",
        markdown=f"""
# Model Evaluation: {model_name}

| Metric | Value |
|--------|-------|
"""
        + "\n".join([f"| {k} | {v:.4f} |" for k, v in metrics.items()]),
        description=f"Evaluation results for {model_name}",
    )

    logger.info(f"Evaluation complete: {metrics}")
    return metrics


@task(
    name="save_model",
    description="Save model to disk",
    tags=["model", "io"],
)
def save_model(
    model: Any,
    model_path: Union[str, Path],
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model to disk.

    Args:
        model: Model to save
        model_path: Base path for models
        model_name: Name of model
        metadata: Optional metadata to save

    Returns:
        Path to saved model
    """
    model_path = Path(model_path)
    save_path = model_path / model_name

    logger.info(f"Saving model to {save_path}")
    model.save(save_path)

    # Save metadata if provided
    if metadata:
        import json

        meta_path = save_path / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Model saved successfully")
    return save_path


@task(
    name="load_model",
    description="Load model from disk",
    tags=["model", "io"],
)
def load_model(
    model_path: Union[str, Path],
    model_type: str,
) -> Any:
    """Load model from disk.

    Args:
        model_path: Path to model directory
        model_type: Type of model

    Returns:
        Loaded model instance
    """
    try:
        from src.models import ChurnPredictor, FraudDetector, PricePredictor
    except ImportError:
        from models import ChurnPredictor, FraudDetector, PricePredictor

    model_classes = {
        "fraud": FraudDetector,
        "price": PricePredictor,
        "churn": ChurnPredictor,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_classes[model_type].load(model_path)
    logger.info(f"Loaded {model_type} model from {model_path}")
    return model


# ============================================================================
# Drift Detection Tasks
# ============================================================================


@task(
    name="check_drift",
    description="Check for data drift",
    tags=["monitoring", "drift"],
)
def check_drift(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    model_name: str,
    psi_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Check for data drift between current and reference data.

    Args:
        current_data: Current data window
        reference_data: Reference (baseline) data
        model_name: Name of model for logging
        psi_threshold: PSI threshold for drift detection

    Returns:
        Drift detection results
    """
    try:
        from src.monitoring.drift_detector import DriftDetector
    except ImportError:
        from monitoring.drift_detector import DriftDetector

    logger.info(f"Checking drift for {model_name}")

    detector = DriftDetector(
        psi_threshold_warning=psi_threshold * 0.5,
        psi_threshold_critical=psi_threshold,
    )

    # Set reference
    detector.set_reference_data(reference_data)

    # Check drift
    result = detector.detect_drift(current_data)

    # Create artifact
    if result.get("drift_detected", False):
        drift_features = result.get("drifted_features", [])
        create_markdown_artifact(
            key=f"{model_name}-drift-alert",
            markdown=f"""
# 🚨 Drift Detected: {model_name}

**Drift Share:** {result.get('drift_share', 0):.2%}

**Drifted Features:**
{chr(10).join([f"- {f}" for f in drift_features])}

**Timestamp:** {datetime.now().isoformat()}
""",
            description=f"Drift alert for {model_name}",
        )

    logger.info(f"Drift check complete: detected={result.get('drift_detected', False)}")
    return result


@task(
    name="compare_distributions",
    description="Compare feature distributions",
    tags=["monitoring", "analysis"],
)
def compare_distributions(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare feature distributions between datasets.

    Args:
        current_data: Current data
        reference_data: Reference data
        feature_columns: Columns to compare (None = all numeric)

    Returns:
        Dictionary of statistics per feature
    """
    import numpy as np

    if feature_columns is None:
        feature_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()

    results = {}
    for col in feature_columns:
        if col in current_data.columns and col in reference_data.columns:
            curr = current_data[col].dropna()
            ref = reference_data[col].dropna()

            results[col] = {
                "current_mean": float(curr.mean()),
                "reference_mean": float(ref.mean()),
                "mean_diff": float(curr.mean() - ref.mean()),
                "current_std": float(curr.std()),
                "reference_std": float(ref.std()),
            }

    return results


# ============================================================================
# Alerting Tasks
# ============================================================================


@task(
    name="send_alert",
    description="Send alert notification",
    tags=["alerting", "notification"],
)
def send_alert(
    alert_type: str,
    message: str,
    severity: str = "warning",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send an alert notification.

    Args:
        alert_type: Type of alert (drift, performance, error)
        message: Alert message
        severity: Alert severity (info, warning, critical)
        metadata: Additional alert metadata

    Returns:
        True if alert sent successfully
    """
    try:
        from src.monitoring.alerts import Alert, AlertManager, AlertSeverity
    except ImportError:
        from monitoring.alerts import Alert, AlertManager, AlertSeverity

    logger.info(f"Sending {severity} alert: {alert_type}")

    severity_map = {
        "info": AlertSeverity.INFO,
        "warning": AlertSeverity.WARNING,
        "critical": AlertSeverity.CRITICAL,
    }

    alert = Alert(
        alert_type=alert_type,
        message=message,
        severity=severity_map.get(severity, AlertSeverity.WARNING),
        metadata=metadata or {},
    )

    # Get global alert manager
    manager = AlertManager()
    manager.add_alert(alert)

    logger.info(f"Alert sent: {alert.id}")
    return True


@task(
    name="check_thresholds",
    description="Check metrics against thresholds",
    tags=["monitoring", "thresholds"],
)
def check_thresholds(
    metrics: Dict[str, float],
    thresholds: Dict[str, Dict[str, float]],
    model_name: str,
) -> List[Dict[str, Any]]:
    """Check metrics against configured thresholds.

    Args:
        metrics: Dictionary of metric values
        thresholds: Dictionary of thresholds per metric
            e.g., {"accuracy": {"min": 0.8, "max": 1.0}}
        model_name: Model name for alerts

    Returns:
        List of threshold violations
    """
    violations = []

    for metric, value in metrics.items():
        if metric in thresholds:
            threshold = thresholds[metric]

            if "min" in threshold and value < threshold["min"]:
                violations.append(
                    {
                        "metric": metric,
                        "value": value,
                        "threshold": threshold["min"],
                        "type": "below_minimum",
                        "model": model_name,
                    }
                )

            if "max" in threshold and value > threshold["max"]:
                violations.append(
                    {
                        "metric": metric,
                        "value": value,
                        "threshold": threshold["max"],
                        "type": "above_maximum",
                        "model": model_name,
                    }
                )

    if violations:
        logger.warning(f"Found {len(violations)} threshold violations for {model_name}")

    return violations
