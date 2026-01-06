"""
Prometheus Metrics Module

Defines and exposes metrics for ML model monitoring.
Tracks:
- Prediction latency and throughput
- Drift scores (PSI, feature drift)
- Model performance metrics
- Data quality metrics
- System health
"""

from typing import Optional

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

# =============================================================================
# Prediction Metrics
# =============================================================================

PREDICTION_COUNTER = Counter(
    "mlobs_predictions_total",
    "Total number of predictions made",
    ["model_name", "model_version", "status"],
)

PREDICTION_LATENCY = Histogram(
    "mlobs_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name", "model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_VALUE = Histogram(
    "mlobs_prediction_value",
    "Distribution of prediction values",
    ["model_name", "model_version"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

BATCH_SIZE = Histogram(
    "mlobs_batch_size",
    "Prediction batch sizes",
    ["model_name"],
    buckets=(1, 10, 50, 100, 500, 1000, 5000, 10000),
)

# =============================================================================
# Drift Metrics
# =============================================================================

DRIFT_SCORE = Gauge(
    "mlobs_drift_score",
    "Current drift score (PSI or similar)",
    ["model_name", "feature_name", "drift_type"],
)

DATASET_DRIFT_DETECTED = Gauge(
    "mlobs_dataset_drift_detected",
    "Whether dataset drift is detected (1=yes, 0=no)",
    ["model_name", "dataset_name"],
)

DRIFT_SHARE = Gauge(
    "mlobs_drift_share",
    "Fraction of features with detected drift",
    ["model_name", "dataset_name"],
)

DRIFTED_FEATURES_COUNT = Gauge(
    "mlobs_drifted_features_count",
    "Number of features with detected drift",
    ["model_name", "dataset_name"],
)

DRIFT_CHECK_COUNTER = Counter(
    "mlobs_drift_checks_total",
    "Total number of drift checks performed",
    ["model_name", "dataset_name", "result"],
)

# =============================================================================
# Model Performance Metrics
# =============================================================================

MODEL_ACCURACY = Gauge(
    "mlobs_model_accuracy",
    "Model accuracy score",
    ["model_name", "model_version", "dataset"],
)

MODEL_PRECISION = Gauge(
    "mlobs_model_precision",
    "Model precision score",
    ["model_name", "model_version", "dataset"],
)

MODEL_RECALL = Gauge(
    "mlobs_model_recall",
    "Model recall score",
    ["model_name", "model_version", "dataset"],
)

MODEL_F1 = Gauge(
    "mlobs_model_f1",
    "Model F1 score",
    ["model_name", "model_version", "dataset"],
)

MODEL_AUC_ROC = Gauge(
    "mlobs_model_auc_roc",
    "Model AUC-ROC score",
    ["model_name", "model_version", "dataset"],
)

MODEL_RMSE = Gauge(
    "mlobs_model_rmse",
    "Model RMSE (regression)",
    ["model_name", "model_version", "dataset"],
)

MODEL_MAE = Gauge(
    "mlobs_model_mae",
    "Model MAE (regression)",
    ["model_name", "model_version", "dataset"],
)

MODEL_R2 = Gauge(
    "mlobs_model_r2",
    "Model R2 score (regression)",
    ["model_name", "model_version", "dataset"],
)

# =============================================================================
# Data Quality Metrics
# =============================================================================

MISSING_VALUES_SHARE = Gauge(
    "mlobs_missing_values_share",
    "Share of missing values in dataset",
    ["model_name", "dataset_name"],
)

DUPLICATE_ROWS_COUNT = Gauge(
    "mlobs_duplicate_rows_count",
    "Number of duplicate rows in dataset",
    ["model_name", "dataset_name"],
)

DATA_QUALITY_SCORE = Gauge(
    "mlobs_data_quality_score",
    "Overall data quality score (0-1)",
    ["model_name", "dataset_name"],
)

FEATURE_MISSING_VALUES = Gauge(
    "mlobs_feature_missing_values",
    "Missing values count per feature",
    ["model_name", "feature_name"],
)

OUTLIER_COUNT = Gauge(
    "mlobs_outlier_count",
    "Number of outliers detected",
    ["model_name", "feature_name"],
)

# =============================================================================
# System Metrics
# =============================================================================

MODEL_LOADED = Gauge(
    "mlobs_model_loaded",
    "Whether model is loaded and ready (1=yes, 0=no)",
    ["model_name", "model_version"],
)

MODEL_INFO = Info(
    "mlobs_model",
    "Model metadata information",
    ["model_name"],
)

LAST_DRIFT_CHECK_TIMESTAMP = Gauge(
    "mlobs_last_drift_check_timestamp",
    "Unix timestamp of last drift check",
    ["model_name"],
)

LAST_TRAINING_TIMESTAMP = Gauge(
    "mlobs_last_training_timestamp",
    "Unix timestamp of last model training",
    ["model_name"],
)

# =============================================================================
# Alert Metrics
# =============================================================================

ALERTS_TOTAL = Counter(
    "mlobs_alerts_total",
    "Total number of alerts generated",
    ["model_name", "alert_type", "severity"],
)

ACTIVE_ALERTS = Gauge(
    "mlobs_active_alerts",
    "Number of currently active alerts",
    ["model_name", "alert_type"],
)


# =============================================================================
# Metric Helper Classes
# =============================================================================


class MetricsCollector:
    """
    Helper class to collect and update metrics.

    Provides a clean interface for updating Prometheus metrics
    from various parts of the application.
    """

    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize the metrics collector.

        Args:
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        logger.debug(f"MetricsCollector initialized for {model_name} v{model_version}")

    def record_prediction(
        self,
        latency_seconds: float,
        prediction_value: Optional[float] = None,
        batch_size: int = 1,
        status: str = "success",
    ) -> None:
        """
        Record a prediction event.

        Args:
            latency_seconds: Time taken for prediction
            prediction_value: The prediction value (for classification probability)
            batch_size: Number of samples in batch
            status: Prediction status (success/error)
        """
        PREDICTION_COUNTER.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            status=status,
        ).inc(batch_size)

        PREDICTION_LATENCY.labels(
            model_name=self.model_name,
            model_version=self.model_version,
        ).observe(latency_seconds)

        if prediction_value is not None:
            PREDICTION_VALUE.labels(
                model_name=self.model_name,
                model_version=self.model_version,
            ).observe(prediction_value)

        BATCH_SIZE.labels(model_name=self.model_name).observe(batch_size)

    def record_drift(
        self,
        dataset_name: str,
        drift_detected: bool,
        drift_share: float,
        drifted_features_count: int,
        feature_scores: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Record drift detection results.

        Args:
            dataset_name: Name of the dataset checked
            drift_detected: Whether drift was detected
            drift_share: Fraction of features with drift
            drifted_features_count: Number of drifted features
            feature_scores: Per-feature drift scores
        """
        DATASET_DRIFT_DETECTED.labels(
            model_name=self.model_name,
            dataset_name=dataset_name,
        ).set(1 if drift_detected else 0)

        DRIFT_SHARE.labels(
            model_name=self.model_name,
            dataset_name=dataset_name,
        ).set(drift_share)

        DRIFTED_FEATURES_COUNT.labels(
            model_name=self.model_name,
            dataset_name=dataset_name,
        ).set(drifted_features_count)

        # Record per-feature drift scores
        if feature_scores:
            for feature_name, score in feature_scores.items():
                DRIFT_SCORE.labels(
                    model_name=self.model_name,
                    feature_name=feature_name,
                    drift_type="psi",
                ).set(score)

        # Update drift check counter
        result = "drift_detected" if drift_detected else "no_drift"
        DRIFT_CHECK_COUNTER.labels(
            model_name=self.model_name,
            dataset_name=dataset_name,
            result=result,
        ).inc()

    def record_performance(
        self,
        metrics: dict[str, float],
        dataset: str = "test",
    ) -> None:
        """
        Record model performance metrics.

        Args:
            metrics: Dictionary of metric names to values
            dataset: Dataset name (train/test/production)
        """
        metric_mapping = {
            "accuracy": MODEL_ACCURACY,
            "precision": MODEL_PRECISION,
            "recall": MODEL_RECALL,
            "f1": MODEL_F1,
            "auc_roc": MODEL_AUC_ROC,
            "rmse": MODEL_RMSE,
            "mae": MODEL_MAE,
            "r2": MODEL_R2,
        }

        for metric_name, gauge in metric_mapping.items():
            # Look for the metric with various prefixes
            for key in [metric_name, f"test_{metric_name}", f"{dataset}_{metric_name}"]:
                if key in metrics:
                    gauge.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        dataset=dataset,
                    ).set(metrics[key])
                    break

    def record_data_quality(
        self,
        dataset_name: str,
        missing_share: float,
        duplicate_count: int,
        quality_score: Optional[float] = None,
        feature_missing: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Record data quality metrics.

        Args:
            dataset_name: Name of the dataset
            missing_share: Share of missing values
            duplicate_count: Number of duplicate rows
            quality_score: Overall quality score (0-1)
            feature_missing: Per-feature missing value counts
        """
        MISSING_VALUES_SHARE.labels(
            model_name=self.model_name,
            dataset_name=dataset_name,
        ).set(missing_share)

        DUPLICATE_ROWS_COUNT.labels(
            model_name=self.model_name,
            dataset_name=dataset_name,
        ).set(duplicate_count)

        if quality_score is not None:
            DATA_QUALITY_SCORE.labels(
                model_name=self.model_name,
                dataset_name=dataset_name,
            ).set(quality_score)

        if feature_missing:
            for feature_name, count in feature_missing.items():
                FEATURE_MISSING_VALUES.labels(
                    model_name=self.model_name,
                    feature_name=feature_name,
                ).set(count)

    def record_alert(
        self,
        alert_type: str,
        severity: str,
    ) -> None:
        """
        Record an alert event.

        Args:
            alert_type: Type of alert (drift, performance, quality)
            severity: Alert severity (warning, critical)
        """
        ALERTS_TOTAL.labels(
            model_name=self.model_name,
            alert_type=alert_type,
            severity=severity,
        ).inc()

    def set_model_loaded(self, loaded: bool = True) -> None:
        """
        Set model loaded status.

        Args:
            loaded: Whether model is loaded
        """
        MODEL_LOADED.labels(
            model_name=self.model_name,
            model_version=self.model_version,
        ).set(1 if loaded else 0)

    def set_last_drift_check(self, timestamp: float) -> None:
        """
        Set timestamp of last drift check.

        Args:
            timestamp: Unix timestamp
        """
        LAST_DRIFT_CHECK_TIMESTAMP.labels(
            model_name=self.model_name,
        ).set(timestamp)


def get_metrics() -> bytes:
    """
    Get all metrics in Prometheus format.

    Returns:
        Metrics in Prometheus text format
    """
    result = generate_latest()
    if isinstance(result, bytes):
        return result
    return bytes(result)


def create_metrics_collector(
    model_name: str,
    model_version: str = "1.0.0",
) -> MetricsCollector:
    """
    Create a metrics collector for a model.

    Args:
        model_name: Name of the model
        model_version: Version of the model

    Returns:
        Configured MetricsCollector instance
    """
    return MetricsCollector(model_name, model_version)
