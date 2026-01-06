"""
Base Model Module

Abstract base class that defines the interface for all ML models.
Provides common functionality for training, prediction, and serialization.
"""

import pickle  # nosec B403
from abc import ABC, abstractmethod
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the platform.

    Provides a consistent interface for:
    - Training and prediction
    - Model serialization/deserialization
    - Performance metrics calculation
    - MLflow integration hooks
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,  # 'classification' or 'regression'
        version: str = "1.0.0",
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name of the model (e.g., 'fraud_detector')
            model_type: Type of model ('classification' or 'regression')
            version: Model version string
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        self.model: Any = None
        self.feature_names: list[str] = []
        self.target_name: str = ""
        self.is_fitted: bool = False
        self.training_date: Optional[datetime] = None
        self.metrics: dict[str, float] = {}
        self.hyperparameters: dict[str, Any] = {}

        logger.info(f"Initialized {model_name} (v{version}) - {model_type}")

    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return the underlying ML model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_default_hyperparameters(self) -> dict[str, Any]:
        """Return default hyperparameters for the model."""
        pass

    def set_hyperparameters(self, **kwargs: Any) -> None:
        """
        Set model hyperparameters.

        Args:
            **kwargs: Hyperparameter key-value pairs
        """
        self.hyperparameters.update(kwargs)
        # Recreate model with new hyperparameters
        self.model = self._create_model()
        logger.debug(f"Updated hyperparameters: {kwargs}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "BaseModel":
        """
        Train the model on the provided data.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional tuple of (X_val, y_val) for validation

        Returns:
            self for method chaining
        """
        logger.info(f"Training {self.model_name} on {len(X)} samples...")

        # Store feature names
        self.feature_names = list(X.columns)
        self.target_name = y.name if hasattr(y, "name") else "target"

        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()

        # Train the model
        self._fit_model(X, y, validation_data)

        self.is_fitted = True
        self.training_date = datetime.now()

        logger.info(f"Training complete for {self.model_name}")
        return self

    def _fit_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> None:
        """
        Internal method to fit the model. Can be overridden for custom training logic.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation_data: Optional validation data
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        self._check_is_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get prediction probabilities (classification only).

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities or None if not supported
        """
        self._check_is_fitted()

        if self.model_type != "classification":
            logger.warning("predict_proba is only available for classification models")
            return None

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "test",
    ) -> dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X: Feature DataFrame
            y: True target values
            prefix: Prefix for metric names (e.g., 'test', 'val')

        Returns:
            Dictionary of metric names to values
        """
        self._check_is_fitted()

        predictions = self.predict(X)
        metrics = {}

        if self.model_type == "classification":
            metrics = self._calculate_classification_metrics(y, predictions, X, prefix)
        else:
            metrics = self._calculate_regression_metrics(y, predictions, prefix)

        self.metrics.update(metrics)
        logger.info(f"Evaluation metrics ({prefix}): {metrics}")
        return metrics

    def _calculate_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X: pd.DataFrame,
        prefix: str,
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{prefix}_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Add AUC-ROC if we can get probabilities
        proba = self.predict_proba(X)
        if proba is not None and len(np.unique(y_true)) == 2:
            with suppress(ValueError, IndexError):
                metrics[f"{prefix}_auc_roc"] = roc_auc_score(y_true, proba[:, 1])

        return metrics

    def _calculate_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        prefix: str,
    ) -> dict[str, float]:
        """Calculate regression metrics."""
        return {
            f"{prefix}_mse": mean_squared_error(y_true, y_pred),
            f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}_r2": r2_score(y_true, y_pred),
        }

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        self._check_is_fitted()

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
            if len(importances) == len(self.feature_names):
                return dict(zip(self.feature_names, importances))
        return None

    def save(self, path: str | Path) -> Path:
        """
        Save model to disk.

        Args:
            path: Directory path to save the model

        Returns:
            Path to saved model file
        """
        self._check_is_fitted()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "training_date": self.training_date,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
        }

        filepath = path / f"{self.model_name}_v{self.version}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str | Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)  # nosec B301

        # Create instance without calling __init__ properly
        instance = cls.__new__(cls)
        instance.model = model_data["model"]
        instance.model_name = model_data["model_name"]
        instance.model_type = model_data["model_type"]
        instance.version = model_data["version"]
        instance.feature_names = model_data["feature_names"]
        instance.target_name = model_data["target_name"]
        instance.training_date = model_data["training_date"]
        instance.metrics = model_data["metrics"]
        instance.hyperparameters = model_data["hyperparameters"]
        instance.is_fitted = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def _check_is_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} is not fitted. Call fit() first.")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model metadata and information.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "is_fitted": self.is_fitted,
            "training_date": (self.training_date.isoformat() if self.training_date else None),
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.model_name}', \
            version='{self.version}', {status})"
