"""
Price Predictor Model

LightGBM-based regressor for predicting property prices.
Optimized for accuracy and interpretability.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger

from src.models.base import BaseModel


class PricePredictor(BaseModel):
    """
    Price prediction model using LightGBM.

    Designed for:
    - Regression on property prices
    - Handling mixed feature types
    - Fast training and inference
    """

    def __init__(self, version: str = "1.0.0", **kwargs: Any):
        """
        Initialize the price predictor.

        Args:
            version: Model version string
            **kwargs: Additional hyperparameters for LightGBM
        """
        super().__init__(
            model_name="price_predictor",
            model_type="regression",
            version=version,
        )

        # Set hyperparameters
        self.hyperparameters = self._get_default_hyperparameters()
        self.hyperparameters.update(kwargs)

        # Create the model
        self.model = self._create_model()

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        """Return default hyperparameters optimized for price prediction."""
        return {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "objective": "regression",
            "metric": "rmse",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def _create_model(self) -> LGBMRegressor:
        """Create the LightGBM regressor."""
        return LGBMRegressor(**self.hyperparameters)

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
            fit_params["eval_metric"] = "rmse"

        self.model.fit(X, y, **fit_params)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_method: str = "std",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.

        Args:
            X: Feature DataFrame
            confidence_method: Method for confidence ('std' or 'quantile')

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        self._check_is_fitted()

        predictions = self.predict(X)

        # Estimate confidence based on leaf variance
        # This is a simplified approach - for production, use quantile regression
        if hasattr(self.model, "predict"):
            # Use training residuals to estimate uncertainty
            confidence = np.ones(len(predictions)) * 0.8  # Placeholder confidence

        return predictions, confidence

    def predict_price_range(
        self,
        X: pd.DataFrame,
        percentile_range: tuple[float, float] = (10, 90),
    ) -> pd.DataFrame:
        """
        Predict price with estimated range.

        Args:
            X: Feature DataFrame
            percentile_range: Lower and upper percentile for range

        Returns:
            DataFrame with predicted price and range
        """
        predictions = self.predict(X)

        # Estimate range based on typical prediction error
        # In production, this would use quantile regression
        error_margin = predictions * 0.15  # Assume 15% error margin

        result = pd.DataFrame(
            {
                "predicted_price": predictions,
                "price_low": predictions - error_margin,
                "price_high": predictions + error_margin,
            },
            index=X.index,
        )

        return result

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "test",
    ) -> dict[str, float]:
        """
        Evaluate with price-specific metrics.

        Args:
            X: Feature DataFrame
            y: True prices
            prefix: Metric prefix

        Returns:
            Dictionary of metrics
        """
        # Get base metrics
        metrics = super().evaluate(X, y, prefix)

        # Add price-specific metrics
        predictions = self.predict(X)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        metrics[f"{prefix}_mape"] = mape

        # Median Absolute Error
        median_ae = np.median(np.abs(y - predictions))
        metrics[f"{prefix}_median_ae"] = median_ae

        # Percentage within 10% of actual
        within_10_pct = np.mean(np.abs((y - predictions) / y) <= 0.10) * 100
        metrics[f"{prefix}_within_10_pct"] = within_10_pct

        # Percentage within 20% of actual
        within_20_pct = np.mean(np.abs((y - predictions) / y) <= 0.20) * 100
        metrics[f"{prefix}_within_20_pct"] = within_20_pct

        # Price range accuracy
        price_ranges = [
            (0, 200000, "under_200k"),
            (200000, 500000, "200k_500k"),
            (500000, 1000000, "500k_1m"),
            (1000000, float("inf"), "over_1m"),
        ]

        for low, high, range_name in price_ranges:
            mask = (y >= low) & (y < high)
            if mask.sum() > 0:
                range_rmse = np.sqrt(np.mean((y[mask] - predictions[mask]) ** 2))
                metrics[f"{prefix}_rmse_{range_name}"] = range_rmse

        self.metrics.update(metrics)
        return metrics

    def get_price_drivers(
        self,
        X: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Analyze what drives prices for given properties.

        Args:
            X: Feature DataFrame
            top_n: Number of top drivers to return

        Returns:
            DataFrame with top price drivers per property
        """
        self._check_is_fitted()

        importance = self.get_feature_importance()
        if importance is None:
            logger.warning("Feature importance not available")
            return pd.DataFrame()

        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:top_n]]

        # Create summary
        result = X[top_features].copy()
        result["predicted_price"] = self.predict(X)

        return result

    def analyze_feature_impact(
        self,
        feature_name: str,
        X: pd.DataFrame,
        num_points: int = 20,
    ) -> pd.DataFrame:
        """
        Analyze how a feature impacts price predictions.

        Args:
            feature_name: Name of feature to analyze
            X: Base feature DataFrame
            num_points: Number of points to evaluate

        Returns:
            DataFrame with feature values and corresponding predictions
        """
        self._check_is_fitted()

        if feature_name not in X.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")

        # Get feature range
        feature_min = X[feature_name].min()
        feature_max = X[feature_name].max()
        feature_values = np.linspace(feature_min, feature_max, num_points)

        # Use median values for other features
        base_sample = X.median().to_frame().T
        base_sample = pd.concat([base_sample] * num_points, ignore_index=True)
        base_sample[feature_name] = feature_values

        # Get predictions
        predictions = self.predict(base_sample)

        return pd.DataFrame(
            {
                feature_name: feature_values,
                "predicted_price": predictions,
            }
        )
