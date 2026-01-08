"""SHAP-based model explainability for ML Observability Platform.

Provides feature importance (global) and local explanations for predictions.
Supports XGBoost, LightGBM, and scikit-learn models.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """Types of SHAP explanations available."""

    TREE = "tree"  # For tree-based models (XGBoost, LightGBM, RandomForest)
    LINEAR = "linear"  # For linear models
    KERNEL = "kernel"  # Model-agnostic (slower)
    DEEP = "deep"  # For neural networks


class SHAPExplainer:
    """SHAP-based explainability for ML models.

    Provides:
    - Global feature importance (mean absolute SHAP values)
    - Local explanations (per-prediction SHAP values)
    - Feature contribution analysis
    - Explanation caching for performance

    Example:
        >>> explainer = SHAPExplainer(model, X_background)
        >>> # Global importance
        >>> importance = explainer.get_feature_importance(X_test)
        >>> # Local explanation for single prediction
        >>> explanation = explainer.explain_prediction(X_test.iloc[[0]])
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[pd.DataFrame] = None,
        explanation_type: ExplanationType = ExplanationType.TREE,
        feature_names: Optional[List[str]] = None,
        max_background_samples: int = 100,
    ):
        """Initialize SHAP explainer.

        Args:
            model: Trained model (XGBoost, LightGBM, RandomForest, etc.)
            background_data: Reference data for SHAP calculations.
                Required for Kernel and Linear explainers.
            explanation_type: Type of SHAP explainer to use.
            feature_names: List of feature names. Auto-detected if not provided.
            max_background_samples: Max samples for background data (for performance).
        """
        # Lazy import to avoid issues if shap not installed
        try:
            import shap
            self._shap = shap
        except ImportError as e:
            raise ImportError(
                "SHAP is required for explainability. "
                "Install with: uv add shap>=0.42.0 numba>=0.56.0"
            ) from e

        self.model = model
        self.explanation_type = explanation_type
        self.feature_names = feature_names
        self.max_background_samples = max_background_samples

        # Prepare background data
        self._background_data = self._prepare_background_data(background_data)

        # Initialize the appropriate explainer
        self._explainer = self._create_explainer()

        # Cache for explanations
        self._explanation_cache: Dict[str, Any] = {}

        logger.info(
            f"Initialized SHAP explainer with type={explanation_type.value}, "
            f"background_samples={len(self._background_data) if self._background_data is not None else 0}"
        )

    def _prepare_background_data(
        self, data: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Prepare and sample background data."""
        if data is None:
            return None

        # Sample if too large
        if len(data) > self.max_background_samples:
            data = data.sample(n=self.max_background_samples, random_state=42)
            logger.info(
                f"Sampled background data to {self.max_background_samples} samples"
            )

        # Auto-detect feature names
        if self.feature_names is None and hasattr(data, "columns"):
            self.feature_names = list(data.columns)

        return data

    def _create_explainer(self) -> Any:
        """Create the appropriate SHAP explainer based on model type."""
        shap = self._shap

        if self.explanation_type == ExplanationType.TREE:
            # Tree explainer for XGBoost, LightGBM, RandomForest, etc.
            try:
                return shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(
                    f"TreeExplainer failed: {e}. Falling back to Explainer."
                )
                return shap.Explainer(self.model)

        elif self.explanation_type == ExplanationType.LINEAR:
            if self._background_data is None:
                raise ValueError("LinearExplainer requires background_data")
            return shap.LinearExplainer(self.model, self._background_data)

        elif self.explanation_type == ExplanationType.KERNEL:
            if self._background_data is None:
                raise ValueError("KernelExplainer requires background_data")
            # Use model's predict method
            predict_fn = (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict
            )
            return shap.KernelExplainer(predict_fn, self._background_data)

        elif self.explanation_type == ExplanationType.DEEP:
            if self._background_data is None:
                raise ValueError("DeepExplainer requires background_data")
            return shap.DeepExplainer(self.model, self._background_data.values)

        else:
            # Default: use SHAP's auto-detection
            return shap.Explainer(self.model)

    def _ensure_numpy(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(X.columns)
            return X.values
        return np.asarray(X)

    def explain_prediction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        include_base_value: bool = True,
    ) -> Dict[str, Any]:
        """Generate local explanation for prediction(s).

        Args:
            X: Input data (single row or multiple rows)
            include_base_value: Include expected value in output

        Returns:
            Dictionary with:
            - shap_values: SHAP values for each feature
            - feature_names: List of feature names
            - feature_values: Input feature values
            - base_value: Expected value (model output with no features)
            - prediction: Model prediction for this input
        """
        X_array = self._ensure_numpy(X)

        # Get SHAP values
        shap_values = self._explainer.shap_values(X_array)

        # Handle multi-output models (e.g., classifiers with probabilities)
        if isinstance(shap_values, list):
            # For binary classification, use positive class (index 1)
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values

        # Get base value (expected value)
        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) == 2 else base_value[0]

        # Get model prediction
        if hasattr(self.model, "predict_proba"):
            prediction = self.model.predict_proba(X_array)
            if prediction.ndim > 1:
                prediction = prediction[:, 1]  # Positive class probability
        else:
            prediction = self.model.predict(X_array)

        result = {
            "shap_values": shap_values.tolist()
            if isinstance(shap_values, np.ndarray)
            else shap_values,
            "feature_names": self.feature_names or [f"feature_{i}" for i in range(X_array.shape[1])],
            "feature_values": X_array.tolist(),
            "prediction": prediction.tolist()
            if isinstance(prediction, np.ndarray)
            else prediction,
        }

        if include_base_value:
            result["base_value"] = float(base_value) if base_value is not None else None

        return result

    def get_feature_importance(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        method: str = "mean_abs",
    ) -> Dict[str, float]:
        """Calculate global feature importance using SHAP values.

        Args:
            X: Data to compute importance on. Uses background data if None.
            method: Aggregation method
                - "mean_abs": Mean absolute SHAP value (default)
                - "mean": Mean SHAP value (can be negative)
                - "max_abs": Maximum absolute SHAP value

        Returns:
            Dictionary mapping feature names to importance scores,
            sorted by importance (descending).
        """
        if X is None:
            if self._background_data is None:
                raise ValueError(
                    "Either provide X or initialize with background_data"
                )
            X = self._background_data

        X_array = self._ensure_numpy(X)

        # Compute SHAP values
        shap_values = self._explainer.shap_values(X_array)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Aggregate by method
        if method == "mean_abs":
            importance = np.abs(shap_values).mean(axis=0)
        elif method == "mean":
            importance = shap_values.mean(axis=0)
        elif method == "max_abs":
            importance = np.abs(shap_values).max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}. Use mean_abs, mean, or max_abs")

        # Create feature importance dictionary
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(importance))
        ]

        importance_dict = dict(zip(feature_names, importance.tolist()))

        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return importance_dict

    def get_feature_contributions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top feature contributions for each prediction.

        Useful for understanding which features drove each prediction.

        Args:
            X: Input data
            top_k: Number of top contributing features to return

        Returns:
            List of dictionaries, one per sample, each containing:
            - positive_contributors: Features pushing prediction up
            - negative_contributors: Features pushing prediction down
            - prediction: Model prediction
        """
        explanation = self.explain_prediction(X, include_base_value=True)
        shap_values = np.array(explanation["shap_values"])
        feature_names = explanation["feature_names"]
        predictions = explanation["prediction"]

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
            predictions = [predictions]

        results = []
        for i, (sv, pred) in enumerate(zip(shap_values, predictions)):
            # Sort features by SHAP value
            sorted_idx = np.argsort(sv)

            # Top positive contributors (highest SHAP values)
            positive_idx = sorted_idx[-top_k:][::-1]
            positive_contributors = [
                {"feature": feature_names[j], "contribution": float(sv[j])}
                for j in positive_idx
                if sv[j] > 0
            ]

            # Top negative contributors (lowest SHAP values)
            negative_idx = sorted_idx[:top_k]
            negative_contributors = [
                {"feature": feature_names[j], "contribution": float(sv[j])}
                for j in negative_idx
                if sv[j] < 0
            ]

            results.append(
                {
                    "sample_index": i,
                    "prediction": pred,
                    "positive_contributors": positive_contributors,
                    "negative_contributors": negative_contributors,
                }
            )

        return results

    def generate_explanation_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """Generate a comprehensive explanation report for a single prediction.

        Args:
            X: Input data
            sample_idx: Index of sample to explain

        Returns:
            Comprehensive explanation dictionary with all details.
        """
        X_array = self._ensure_numpy(X)

        if sample_idx >= len(X_array):
            raise ValueError(f"sample_idx {sample_idx} out of range for data of size {len(X_array)}")

        # Get single sample
        X_single = X_array[[sample_idx]]

        # Get explanation
        explanation = self.explain_prediction(X_single)

        # Get contributions
        contributions = self.get_feature_contributions(X_single, top_k=10)[0]

        # Calculate summary stats
        shap_values = np.array(explanation["shap_values"]).flatten()
        feature_names = explanation["feature_names"]
        feature_values = np.array(explanation["feature_values"]).flatten()

        # Build report
        report = {
            "prediction": explanation["prediction"][0] if isinstance(explanation["prediction"], list) else explanation["prediction"],
            "base_value": explanation.get("base_value"),
            "total_shap_contribution": float(np.sum(shap_values)),
            "feature_details": [
                {
                    "name": feature_names[i],
                    "value": float(feature_values[i]),
                    "shap_value": float(shap_values[i]),
                    "abs_importance_rank": int(
                        np.argsort(np.argsort(np.abs(shap_values)[::-1]))[i] + 1
                    ),
                }
                for i in range(len(feature_names))
            ],
            "top_positive_contributors": contributions["positive_contributors"][:5],
            "top_negative_contributors": contributions["negative_contributors"][:5],
            "explanation_type": self.explanation_type.value,
        }

        return report

    def to_dict(self) -> Dict[str, Any]:
        """Serialize explainer metadata to dictionary."""
        return {
            "explanation_type": self.explanation_type.value,
            "feature_names": self.feature_names,
            "max_background_samples": self.max_background_samples,
            "has_background_data": self._background_data is not None,
        }


class ModelExplainerRegistry:
    """Registry for managing explainers for multiple models."""

    def __init__(self):
        self._explainers: Dict[str, SHAPExplainer] = {}

    def register(
        self,
        model_name: str,
        model: Any,
        background_data: Optional[pd.DataFrame] = None,
        explanation_type: ExplanationType = ExplanationType.TREE,
    ) -> SHAPExplainer:
        """Register a model for explainability.

        Args:
            model_name: Unique identifier for the model
            model: Trained model instance
            background_data: Reference data for SHAP
            explanation_type: Type of explainer

        Returns:
            Created SHAPExplainer instance
        """
        explainer = SHAPExplainer(
            model=model,
            background_data=background_data,
            explanation_type=explanation_type,
        )
        self._explainers[model_name] = explainer
        logger.info(f"Registered explainer for model: {model_name}")
        return explainer

    def get(self, model_name: str) -> Optional[SHAPExplainer]:
        """Get explainer for a model."""
        return self._explainers.get(model_name)

    def explain(
        self,
        model_name: str,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Any]:
        """Get explanation for a prediction using registered explainer."""
        explainer = self.get(model_name)
        if explainer is None:
            raise ValueError(f"No explainer registered for model: {model_name}")
        return explainer.explain_prediction(X)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._explainers.keys())


# Global registry instance
explainer_registry = ModelExplainerRegistry()
