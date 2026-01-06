"""
Churn Predictor Model

Random Forest-based classifier for predicting customer churn.
Optimized for interpretability and balanced performance.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel


class ChurnPredictor(BaseModel):
    """
    Churn prediction model using Random Forest.

    Designed for:
    - Binary classification (churn/no-churn)
    - Interpretable feature importance
    - Handling mixed feature types
    """

    def __init__(self, version: str = "1.0.0", **kwargs: Any):
        """
        Initialize the churn predictor.

        Args:
            version: Model version string
            **kwargs: Additional hyperparameters for Random Forest
        """
        super().__init__(
            model_name="churn_predictor",
            model_type="classification",
            version=version,
        )

        # Set hyperparameters
        self.hyperparameters = self._get_default_hyperparameters()
        self.hyperparameters.update(kwargs)

        # Create the model
        self.model = self._create_model()

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        """Return default hyperparameters optimized for churn prediction."""
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",  # Handle class imbalance
            "random_state": 42,
            "n_jobs": -1,
        }

    def _create_model(self) -> RandomForestClassifier:
        """Create the Random Forest classifier."""
        return RandomForestClassifier(**self.hyperparameters)

    def predict_churn_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get churn probability scores.

        Args:
            X: Feature DataFrame

        Returns:
            Array of churn probabilities (0-1)
        """
        proba = self.predict_proba(X)
        if proba is not None:
            return proba[:, 1]  # Probability of churn (class 1)
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
        probabilities = self.predict_churn_probability(X)
        return (probabilities >= threshold).astype(int)

    def get_at_risk_customers(
        self,
        X: pd.DataFrame,
        customer_ids: Optional[pd.Series] = None,
        threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        Identify customers at risk of churning.

        Args:
            X: Feature DataFrame
            customer_ids: Optional customer ID series
            threshold: Minimum probability to flag as at-risk

        Returns:
            DataFrame with at-risk customers and their scores
        """
        probabilities = self.predict_churn_probability(X)
        at_risk_mask = probabilities >= threshold

        result = X[at_risk_mask].copy()
        result["churn_probability"] = probabilities[at_risk_mask]
        result["risk_level"] = pd.cut(
            result["churn_probability"],
            bins=[0, 0.6, 0.75, 0.9, 1.0],
            labels=["moderate", "high", "very_high", "critical"],
        )

        if customer_ids is not None:
            result["customer_id"] = customer_ids[at_risk_mask].values

        return result.sort_values("churn_probability", ascending=False)

    def segment_customers(
        self,
        X: pd.DataFrame,
        customer_ids: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Segment customers by churn risk.

        Args:
            X: Feature DataFrame
            customer_ids: Optional customer ID series

        Returns:
            DataFrame with customer segments
        """
        probabilities = self.predict_churn_probability(X)

        result = X.copy()
        result["churn_probability"] = probabilities
        result["segment"] = pd.cut(
            probabilities,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["loyal", "stable", "watch", "at_risk", "likely_churn"],
        )

        if customer_ids is not None:
            result["customer_id"] = customer_ids.values

        return result

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "test",
    ) -> dict[str, float]:
        """
        Evaluate with churn-specific metrics.

        Args:
            X: Feature DataFrame
            y: True labels
            prefix: Metric prefix

        Returns:
            Dictionary of metrics
        """
        # Get base metrics
        metrics = super().evaluate(X, y, prefix)

        # Add churn-specific metrics
        predictions = self.predict(X)
        probabilities = self.predict_churn_probability(X)

        # Confusion matrix components
        tp = ((predictions == 1) & (y == True)).sum()  # noqa: E712
        fp = ((predictions == 1) & (y == False)).sum()  # noqa: E712
        fn = ((predictions == 0) & (y == True)).sum()  # noqa: E712
        # tn = ((predictions == 0) & (y == False)).sum()  # noqa: E712

        # Churn detection rate (recall for churn class)
        if (tp + fn) > 0:
            metrics[f"{prefix}_churn_detection_rate"] = tp / (tp + fn)

        # Retention rate if we act on predictions
        # Assuming we can retain 50% of flagged customers
        potential_saves = tp * 0.5
        total_churners = tp + fn
        if total_churners > 0:
            metrics[f"{prefix}_potential_retention_rate"] = potential_saves / total_churners

        # Cost-based metrics (example: cost of false positive vs false negative)
        # FN cost: losing a customer (high cost)
        # FP cost: unnecessary retention effort (low cost)
        fn_cost = 100  # Cost of losing a customer
        fp_cost = 10  # Cost of unnecessary retention effort
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        metrics[f"{prefix}_total_cost"] = total_cost

        # Lift at different percentiles
        for percentile in [10, 20, 30]:
            n_samples = int(len(y) * percentile / 100)
            if n_samples > 0:
                top_indices = np.argsort(probabilities)[-n_samples:]
                actual_churn_in_top = y.iloc[top_indices].sum()
                random_expected = n_samples * y.mean()
                if random_expected > 0:
                    lift = actual_churn_in_top / random_expected
                    metrics[f"{prefix}_lift_at_{percentile}pct"] = lift

        self.metrics.update(metrics)
        return metrics

    def get_retention_recommendations(
        self,
        X: pd.DataFrame,
        customer_ids: Optional[pd.Series] = None,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Get retention recommendations for at-risk customers.

        Args:
            X: Feature DataFrame
            customer_ids: Optional customer ID series
            top_n: Number of top customers to return

        Returns:
            DataFrame with recommendations
        """
        probabilities = self.predict_churn_probability(X)
        # importance = self.get_feature_importance()

        # Get top at-risk customers
        top_indices = np.argsort(probabilities)[-top_n:][::-1]

        result = X.iloc[top_indices].copy()
        result["churn_probability"] = probabilities[top_indices]

        if customer_ids is not None:
            result["customer_id"] = customer_ids.iloc[top_indices].values

        # Add simple recommendations based on feature values
        recommendations = []
        for idx in top_indices:
            rec = []
            row = X.iloc[idx]

            # Check key churn indicators
            if "login_frequency" in X.columns and row.get("login_frequency", 999) < 2:
                rec.append("Increase engagement - low login frequency")
            if "support_tickets" in X.columns and row.get("support_tickets", 0) > 3:
                rec.append("Address support issues - high ticket count")
            if "last_activity_days" in X.columns and row.get("last_activity_days", 0) > 14:
                rec.append("Re-engagement campaign - inactive user")
            if "nps_score" in X.columns and row.get("nps_score", 10) < 5:
                rec.append("Satisfaction follow-up - low NPS")
            if "contract_type" in X.columns and row.get("contract_type") == "month-to-month":
                rec.append("Offer annual contract discount")

            recommendations.append("; ".join(rec) if rec else "General retention outreach")

        result["recommendations"] = recommendations
        result["priority"] = range(1, len(result) + 1)

        return result

    def analyze_churn_factors(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Analyze factors that contribute to churn.

        Args:
            X: Feature DataFrame
            y: Churn labels

        Returns:
            DataFrame with factor analysis
        """
        self._check_is_fitted()

        churned = X[y == True]  # noqa: E712
        retained = X[y == False]  # noqa: E712

        analysis = []
        for col in X.select_dtypes(include=[np.number]).columns:
            churned_mean = churned[col].mean()
            retained_mean = retained[col].mean()
            diff_pct = (
                ((churned_mean - retained_mean) / retained_mean * 100) if retained_mean != 0 else 0
            )

            analysis.append(
                {
                    "feature": col,
                    "churned_mean": churned_mean,
                    "retained_mean": retained_mean,
                    "difference_pct": diff_pct,
                    "impact": (
                        "increases_churn"
                        if diff_pct > 10
                        else ("decreases_churn" if diff_pct < -10 else "neutral")
                    ),
                }
            )

        result = pd.DataFrame(analysis)
        result = result.sort_values("difference_pct", key=abs, ascending=False)

        return result
