"""
Prediction Routes

Provides endpoints for making predictions with all ML models.
"""

import time
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    FraudPredictionRequest,
    FraudPredictionResponse,
    PricePredictionRequest,
    PricePredictionResponse,
)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Global model storage (will be set by app startup)
_models: dict[str, Any] = {
    "fraud": None,
    "price": None,
    "churn": None,
}

_preprocessors: dict[str, Any] = {
    "fraud": None,
    "price": None,
    "churn": None,
}

# ============================================================================
# FRAUD MODEL - Feature configurations
# ============================================================================

FRAUD_EXPECTED_FEATURES = [
    'amount', 'latitude', 'longitude', 'distance_from_home', 'hour_of_day',
    'day_of_week', 'avg_transaction_amount', 'transaction_count_24h',
    'transaction_count_7d', 'transaction_type_deposit', 'transaction_type_payment',
    'transaction_type_purchase', 'transaction_type_transfer', 'transaction_type_withdrawal',
    'merchant_category_entertainment', 'merchant_category_gas_station',
    'merchant_category_grocery', 'merchant_category_healthcare',
    'merchant_category_online_shopping', 'merchant_category_other',
    'merchant_category_restaurant', 'merchant_category_retail',
    'merchant_category_travel', 'merchant_category_utilities',
    'is_weekend', 'is_online', 'is_foreign'
]

FRAUD_TRANSACTION_TYPES = ['deposit', 'payment', 'purchase', 'transfer', 'withdrawal']
FRAUD_MERCHANT_CATEGORIES = [
    'entertainment', 'gas_station', 'grocery', 'healthcare',
    'online_shopping', 'other', 'restaurant', 'retail', 'travel', 'utilities'
]


def prepare_fraud_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare fraud features with one-hot encoding to match trained model."""
    df = data.copy()
    
    transaction_type = str(df['transaction_type'].iloc[0]).lower() if 'transaction_type' in df.columns else 'purchase'
    merchant_category = str(df['merchant_category'].iloc[0]).lower() if 'merchant_category' in df.columns else 'retail'
    
    result = pd.DataFrame(index=df.index)
    
    numerical_cols = [
        'amount', 'latitude', 'longitude', 'distance_from_home', 'hour_of_day',
        'day_of_week', 'avg_transaction_amount', 'transaction_count_24h',
        'transaction_count_7d', 'is_weekend', 'is_online', 'is_foreign'
    ]
    
    for col in numerical_cols:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = 0
    
    for tt in FRAUD_TRANSACTION_TYPES:
        col_name = f'transaction_type_{tt}'
        result[col_name] = 1 if transaction_type == tt else 0
    
    for mc in FRAUD_MERCHANT_CATEGORIES:
        col_name = f'merchant_category_{mc}'
        mc_match = merchant_category == mc or merchant_category.replace('_', '') == mc.replace('_', '')
        if merchant_category == 'online':
            mc_match = mc == 'online_shopping'
        result[col_name] = 1 if mc_match else 0
    
    result = result.reindex(columns=FRAUD_EXPECTED_FEATURES, fill_value=0)
    return result


# ============================================================================
# PRICE MODEL - Feature configurations
# ============================================================================

PRICE_PROPERTY_TYPES = ['apartment', 'condo', 'house', 'townhouse']

PRICE_NUMERICAL_COLS = [
    'square_feet', 'bedrooms', 'bathrooms', 'year_built', 'latitude', 'longitude',
    'neighborhood_score', 'school_rating', 'crime_rate', 'has_garage', 'has_pool',
    'has_garden', 'renovated', 'days_on_market', 'num_price_changes'
]


def prepare_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare price features with one-hot encoding to match trained model."""
    df = data.copy()
    
    property_type = str(df['property_type'].iloc[0]).lower().replace(' ', '_') if 'property_type' in df.columns else 'house'
    if property_type == 'single_family':
        property_type = 'house'
    
    result = pd.DataFrame(index=df.index)
    
    for col in PRICE_NUMERICAL_COLS:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = 0
    
    for pt in PRICE_PROPERTY_TYPES:
        col_name = f'property_type_{pt}'
        result[col_name] = 1 if property_type == pt else 0
    
    return result


# ============================================================================
# CHURN MODEL - Feature configurations (EXACT ORDER from model)
# ============================================================================

# Exact feature order from model.feature_names_in_
CHURN_EXPECTED_FEATURES = [
    'age', 'monthly_charges', 'total_charges', 'tenure_months', 'login_frequency',
    'feature_usage_score', 'last_activity_days', 'support_tickets', 'complaints',
    'referrals', 'nps_score', 'gender_female', 'gender_male', 'gender_other',
    'location_midwest', 'location_northeast', 'location_southeast', 'location_southwest',
    'location_west', 'subscription_plan_basic', 'subscription_plan_enterprise',
    'subscription_plan_free', 'subscription_plan_premium', 'payment_method_bank_transfer',
    'payment_method_credit_card', 'payment_method_debit_card', 'payment_method_paypal',
    'contract_type_annual', 'contract_type_month-to-month', 'email_opt_in', 'auto_renewal'
]

CHURN_GENDERS = ['female', 'male', 'other']
CHURN_LOCATIONS = ['midwest', 'northeast', 'southeast', 'southwest', 'west']
CHURN_SUBSCRIPTION_PLANS = ['basic', 'enterprise', 'free', 'premium']
CHURN_PAYMENT_METHODS = ['bank_transfer', 'credit_card', 'debit_card', 'paypal']
CHURN_CONTRACT_TYPES = ['annual', 'month-to-month']


def prepare_churn_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare churn features with one-hot encoding in EXACT order model expects."""
    df = data.copy()
    
    # Get categorical values and normalize
    gender = str(df['gender'].iloc[0]).lower() if 'gender' in df.columns else 'male'
    location = str(df['location'].iloc[0]).lower() if 'location' in df.columns else 'midwest'
    subscription_plan = str(df['subscription_plan'].iloc[0]).lower() if 'subscription_plan' in df.columns else 'basic'
    payment_method = str(df['payment_method'].iloc[0]).lower().replace(' ', '_') if 'payment_method' in df.columns else 'credit_card'
    contract_type = str(df['contract_type'].iloc[0]).lower().replace(' ', '-').replace('_', '-') if 'contract_type' in df.columns else 'month-to-month'
    
    # Normalize contract_type variations
    if contract_type in ['monthly', 'month-to-month']:
        contract_type = 'month-to-month'
    elif contract_type in ['yearly', 'one-year', 'two-year', 'annual']:
        contract_type = 'annual'
    
    # Map location if using different naming
    location_map = {
        'urban': 'northeast',
        'suburban': 'midwest', 
        'rural': 'southwest'
    }
    if location in location_map:
        location = location_map[location]
    
    result = pd.DataFrame(index=df.index)
    
    # Numerical columns (in order)
    numerical_cols = [
        'age', 'monthly_charges', 'total_charges', 'tenure_months', 'login_frequency',
        'feature_usage_score', 'last_activity_days', 'support_tickets', 'complaints',
        'referrals', 'nps_score'
    ]
    for col in numerical_cols:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = 0
    
    # One-hot encode gender (in order)
    for g in CHURN_GENDERS:
        col_name = f'gender_{g}'
        result[col_name] = 1 if gender == g else 0
    
    # One-hot encode location (in order)
    for loc in CHURN_LOCATIONS:
        col_name = f'location_{loc}'
        result[col_name] = 1 if location == loc else 0
    
    # One-hot encode subscription_plan (in order)
    for sp in CHURN_SUBSCRIPTION_PLANS:
        col_name = f'subscription_plan_{sp}'
        result[col_name] = 1 if subscription_plan == sp else 0
    
    # One-hot encode payment_method (in order)
    for pm in CHURN_PAYMENT_METHODS:
        col_name = f'payment_method_{pm}'
        result[col_name] = 1 if payment_method == pm else 0
    
    # One-hot encode contract_type (in order)
    for ct in CHURN_CONTRACT_TYPES:
        col_name = f'contract_type_{ct}'
        result[col_name] = 1 if contract_type == ct else 0
    
    # Boolean columns at the end
    result['email_opt_in'] = df['email_opt_in'].iloc[0] if 'email_opt_in' in df.columns else 0
    result['auto_renewal'] = df['auto_renewal'].iloc[0] if 'auto_renewal' in df.columns else 0
    
    # Reorder to exact expected order
    result = result.reindex(columns=CHURN_EXPECTED_FEATURES, fill_value=0)
    
    return result


# ============================================================================
# Helper functions
# ============================================================================

def set_model(model_type: str, model: Any, preprocessor: Any = None) -> None:
    """Set a model for predictions."""
    _models[model_type] = model
    if preprocessor:
        _preprocessors[model_type] = preprocessor
    logger.info(f"Model set: {model_type}")


def get_model(model_type: str) -> Any:
    """Get a model by type."""
    return _models.get(model_type)


def get_preprocessor(model_type: str) -> Any:
    """Get a preprocessor by type."""
    return _preprocessors.get(model_type)


def _generate_prediction_id() -> str:
    """Generate a unique prediction ID."""
    return str(uuid.uuid4())


def _get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability < 0.3:
        return "low"
    elif probability < 0.5:
        return "medium"
    elif probability < 0.7:
        return "high"
    else:
        return "critical"


def _get_risk_segment(probability: float) -> str:
    """Convert churn probability to risk segment."""
    if probability < 0.2:
        return "loyal"
    elif probability < 0.4:
        return "stable"
    elif probability < 0.6:
        return "at_risk"
    elif probability < 0.8:
        return "high_risk"
    else:
        return "critical"


def _get_retention_priority(probability: float) -> int:
    """Convert churn probability to retention priority (1-5)."""
    if probability < 0.2:
        return 5
    elif probability < 0.4:
        return 4
    elif probability < 0.6:
        return 3
    elif probability < 0.8:
        return 2
    else:
        return 1


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/fraud",
    response_model=FraudPredictionResponse,
    summary="Fraud Prediction",
    description="Predict whether a transaction is fraudulent.",
)
async def predict_fraud(request: FraudPredictionRequest) -> FraudPredictionResponse:
    """Make a fraud prediction for a single transaction."""
    start_time = time.time()

    model = get_model("fraud")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection model not loaded",
        )

    try:
        data = pd.DataFrame([request.model_dump()])
        data = prepare_fraud_features(data)

        prediction = model.predict(data)[0]
        probability = model.predict_fraud_probability(data)[0]

        latency_ms = (time.time() - start_time) * 1000

        return FraudPredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_level=_get_risk_level(probability),
            model_version=model.version,
            prediction_id=_generate_prediction_id(),
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Fraud prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/price",
    response_model=PricePredictionResponse,
    summary="Price Prediction",
    description="Predict the price of a property.",
)
async def predict_price(request: PricePredictionRequest) -> PricePredictionResponse:
    """Make a price prediction for a single property."""
    start_time = time.time()

    model = get_model("price")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Price prediction model not loaded",
        )

    try:
        data = pd.DataFrame([request.model_dump()])
        data = prepare_price_features(data)

        prediction = model.predict(data)[0]

        price_range = prediction * 0.15
        price_low = prediction - price_range
        price_high = prediction + price_range

        latency_ms = (time.time() - start_time) * 1000

        return PricePredictionResponse(
            predicted_price=float(prediction),
            price_range_low=float(price_low),
            price_range_high=float(price_high),
            confidence=0.85,
            model_version=model.version,
            prediction_id=_generate_prediction_id(),
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Price prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/churn",
    response_model=ChurnPredictionResponse,
    summary="Churn Prediction",
    description="Predict whether a customer will churn.",
)
async def predict_churn(request: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Make a churn prediction for a single customer."""
    start_time = time.time()

    model = get_model("churn")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn prediction model not loaded",
        )

    try:
        data = pd.DataFrame([request.model_dump()])
        data = prepare_churn_features(data)

        prediction = model.predict(data)[0]
        probability = model.predict_churn_probability(data)[0]

        latency_ms = (time.time() - start_time) * 1000

        return ChurnPredictionResponse(
            will_churn=bool(prediction),
            churn_probability=float(probability),
            risk_segment=_get_risk_segment(probability),
            retention_priority=_get_retention_priority(probability),
            model_version=model.version,
            prediction_id=_generate_prediction_id(),
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Churn prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Prediction",
    description="Make predictions for multiple instances.",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Make predictions for a batch of instances."""
    start_time = time.time()

    model_type = request.model_type.value
    model = get_model(model_type)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{model_type} model not loaded",
        )

    try:
        data = pd.DataFrame(request.instances)

        # Prepare features based on model type
        if model_type == "fraud":
            processed_rows = [prepare_fraud_features(data.iloc[[idx]]) for idx in range(len(data))]
            data = pd.concat(processed_rows, ignore_index=True)
        elif model_type == "price":
            processed_rows = [prepare_price_features(data.iloc[[idx]]) for idx in range(len(data))]
            data = pd.concat(processed_rows, ignore_index=True)
        elif model_type == "churn":
            processed_rows = [prepare_churn_features(data.iloc[[idx]]) for idx in range(len(data))]
            data = pd.concat(processed_rows, ignore_index=True)

        predictions_list = []

        if model_type == "fraud":
            preds = model.predict(data)
            probs = model.predict_fraud_probability(data)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                predictions_list.append({
                    "index": i,
                    "is_fraud": bool(pred),
                    "fraud_probability": float(prob),
                    "risk_level": _get_risk_level(prob),
                })

        elif model_type == "price":
            preds = model.predict(data)
            for i, pred in enumerate(preds):
                predictions_list.append({
                    "index": i,
                    "predicted_price": float(pred),
                    "price_range_low": float(pred * 0.85),
                    "price_range_high": float(pred * 1.15),
                })

        elif model_type == "churn":
            preds = model.predict(data)
            probs = model.predict_churn_probability(data)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                predictions_list.append({
                    "index": i,
                    "will_churn": bool(pred),
                    "churn_probability": float(prob),
                    "risk_segment": _get_risk_segment(prob),
                })

        total_latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(request.instances)

        return BatchPredictionResponse(
            predictions=predictions_list,
            model_type=request.model_type,
            model_version=model.version,
            total_instances=len(request.instances),
            total_latency_ms=total_latency_ms,
            avg_latency_ms=avg_latency_ms,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.get(
    "/models",
    summary="List Available Models",
    description="Get list of available models and their status.",
)
async def list_models() -> dict[str, Any]:
    """List all available models and their loading status."""
    models_info = {}

    for model_type, model in _models.items():
        if model is not None:
            models_info[model_type] = {
                "loaded": True,
                "version": model.version,
                "model_name": model.model_name,
                "model_type": model.model_type,
            }
        else:
            models_info[model_type] = {
                "loaded": False,
                "version": None,
                "model_name": None,
                "model_type": None,
            }

    return {"models": models_info}