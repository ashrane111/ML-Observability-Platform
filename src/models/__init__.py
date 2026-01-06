"""
Models Module

Provides ML models for the observability platform:
- FraudDetector: XGBoost classifier for fraud detection
- PricePredictor: LightGBM regressor for price prediction
- ChurnPredictor: Random Forest classifier for churn prediction
"""

from src.models.base import BaseModel
from src.models.churn_predictor import ChurnPredictor
from src.models.fraud_detector import FraudDetector
from src.models.preprocessing import (
    FeaturePreprocessor,
    prepare_churn_features,
    prepare_fraud_features,
    prepare_price_features,
)
from src.models.price_predictor import PricePredictor

__all__ = [
    # Base
    "BaseModel",
    # Models
    "FraudDetector",
    "PricePredictor",
    "ChurnPredictor",
    # Preprocessing
    "FeaturePreprocessor",
    "prepare_fraud_features",
    "prepare_price_features",
    "prepare_churn_features",
]
