"""
Fraud Detector Model

XGBoost-based classifier for detecting fraudulent transactions.
Optimized for imbalanced datasets with high precision requirements.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from src.models.base import BaseModel


class FraudDetector(BaseModel):
    """
    Fraud detection model using XGBoost.

    Designed for:
    - Highly imbalanced datasets (low fraud rate)
    - High precision requirements (minimize false positives)
    - Real-time prediction capability
    """

    def __init__(self, version: str = "1.0.0", **kwargs: Any):
        """
        Initialize the fraud detector.

        Args:
            version: Model version string
            **kwargs: Additional hyperparameters for XGBoost
        """
        super().__init__(
            model_name="fraud_detector",
            model_type="classification",
            version=version,
        )

        # Set hyperparameters
        self.hyperparameters = self._get_default_hyperparameters()
        self.hyperparameters.update(kwargs)

        # Create the model
        self.model = self._create_model()

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        """Return default hyperparameters optimized for fraud detection."""
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,  # Will be adjusted for imbalance
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        }

    def _create_model(self) -> XGBClassifier:
        """Create the XGBoost classifier."""
        return XGBClassifier(**self.hyperparameters)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
        auto_balance: bool = True,
    ) -> "FraudDetector":
        """
        Train the fraud detector.

        Args:
            X: Feature DataFrame
            y: Target Series (binary: True/False for fraud)
            validation_data: Optional validation data tuple
            auto_balance: Whether to auto-adjust for class imbalance

        Returns:
            self for method chaining
        """
        # Handle class imbalance
        if auto_balance:
            neg_count = (y == False).sum()  # noqa: E712
            pos_count = (y == True).sum()  # noqa: E712
            if pos_count > 0:
                scale_pos_weight = neg_count / pos_count
                self.hyperparameters["scale_pos_weight"] = scale_pos_weight
                self.model = self._create_model()
                logger.info(f"Auto-balanced: scale_pos_weight={scale_pos_weight:.2f}")
        super().fit(X, y, validation_data)
        return self

    def _fit_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> None:
        """Fit with optional early stopping."""
        fit_params: dict[str, Any] = {}

        if validation_data is not None:
            X_val, y_val = validation_data
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)

    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get fraud probability scores.

        Args:
            X: Feature DataFrame

        Returns:
            Array of fraud probabilities (0-1)
        """
        proba = self.predict_proba(X)
        if proba is not None:
            return proba[:, 1]  # Probability of fraud (class 1)
        return self.predict(X).astype(float)

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Make predictions with custom threshold.

        Args:
            X: Feature DataFrame
            threshold: Classification threshold (default 0.5)

        Returns:
            Binary predictions
        """
        probabilities = self.predict_fraud_probability(X)
        return (probabilities >= threshold).astype(int)

    def get_high_risk_transactions(
        self,
        X: pd.DataFrame,
        threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        Identify high-risk transactions.

        Args:
            X: Feature DataFrame
            threshold: Minimum probability to flag as high risk

        Returns:
            DataFrame with high-risk transactions and their scores
        """
        probabilities = self.predict_fraud_probability(X)
        high_risk_mask = probabilities >= threshold

        result = X[high_risk_mask].copy()
        result["fraud_probability"] = probabilities[high_risk_mask]
        result["risk_level"] = pd.cut(
            result["fraud_probability"],
            bins=[0, 0.7, 0.85, 1.0],
            labels=["high", "very_high", "critical"],
        )

        return result.sort_values("fraud_probability", ascending=False)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "test",
    ) -> dict[str, float]:
        """
        Evaluate with fraud-specific metrics.

        Args:
            X: Feature DataFrame
            y: True labels
            prefix: Metric prefix

        Returns:
            Dictionary of metrics
        """
        # Get base metrics
        metrics = super().evaluate(X, y, prefix)

        # Add fraud-specific metrics
        predictions = self.predict(X)
        probabilities = self.predict_fraud_probability(X)

        # True positives, false positives, etc.
        tp = ((predictions == 1) & (y == True)).sum()  # noqa: E712
        fp = ((predictions == 1) & (y == False)).sum()  # noqa: E712
        fn = ((predictions == 0) & (y == True)).sum()  # noqa: E712
        tn = ((predictions == 0) & (y == False)).sum()  # noqa: E712

        # Fraud detection rate (recall for fraud class)
        if (tp + fn) > 0:
            metrics[f"{prefix}_fraud_detection_rate"] = tp / (tp + fn)

        # False positive rate
        if (fp + tn) > 0:
            metrics[f"{prefix}_false_positive_rate"] = fp / (fp + tn)

        # Precision at different thresholds
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            preds_at_threshold = (probabilities >= threshold).astype(int)
            tp_t = ((preds_at_threshold == 1) & (y == True)).sum()  # noqa: E712
            fp_t = ((preds_at_threshold == 1) & (y == False)).sum()  # noqa: E712
            if (tp_t + fp_t) > 0:
                precision_t = tp_t / (tp_t + fp_t)
                metrics[f"{prefix}_precision_at_{int(threshold*100)}"] = precision_t

        self.metrics.update(metrics)
        return metrics

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """Get feature importance with gain and weight metrics."""
        self._check_is_fitted()

        importance_dict = {}

        # Gain-based importance (default)
        if hasattr(self.model, "feature_importances_"):
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[name] = float(importance)

        return importance_dict if importance_dict else None
