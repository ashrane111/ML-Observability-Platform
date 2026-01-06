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
        return 5  # Lowest priority
    elif probability < 0.4:
        return 4
    elif probability < 0.6:
        return 3
    elif probability < 0.8:
        return 2
    else:
        return 1  # Highest priority


@router.post(
    "/fraud",
    response_model=FraudPredictionResponse,
    summary="Fraud Prediction",
    description="Predict whether a transaction is fraudulent.",
)
async def predict_fraud(request: FraudPredictionRequest) -> FraudPredictionResponse:
    """
    Make a fraud prediction for a single transaction.

    Returns fraud probability and risk level.
    """
    start_time = time.time()

    model = get_model("fraud")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection model not loaded",
        )

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Preprocess if preprocessor available
        preprocessor = get_preprocessor("fraud")
        if preprocessor:
            data = preprocessor.transform(data)

        # Make prediction
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
    """
    Make a price prediction for a single property.

    Returns predicted price with confidence interval.
    """
    start_time = time.time()

    model = get_model("price")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Price prediction model not loaded",
        )

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Preprocess if preprocessor available
        preprocessor = get_preprocessor("price")
        if preprocessor:
            data = preprocessor.transform(data)

        # Make prediction
        prediction = model.predict(data)[0]

        # Calculate price range (±15% as estimate)
        price_range = prediction * 0.15
        price_low = prediction - price_range
        price_high = prediction + price_range

        latency_ms = (time.time() - start_time) * 1000

        return PricePredictionResponse(
            predicted_price=float(prediction),
            price_range_low=float(price_low),
            price_range_high=float(price_high),
            confidence=0.85,  # Placeholder confidence
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
    """
    Make a churn prediction for a single customer.

    Returns churn probability and risk segment.
    """
    start_time = time.time()

    model = get_model("churn")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn prediction model not loaded",
        )

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Preprocess if preprocessor available
        preprocessor = get_preprocessor("churn")
        if preprocessor:
            data = preprocessor.transform(data)

        # Make prediction
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
    """
    Make predictions for a batch of instances.

    Supports all model types: fraud, price, churn.
    """
    start_time = time.time()

    model_type = request.model_type.value
    model = get_model(model_type)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{model_type} model not loaded",
        )

    try:
        # Convert to DataFrame
        data = pd.DataFrame(request.instances)

        # Preprocess if preprocessor available
        preprocessor = get_preprocessor(model_type)
        if preprocessor:
            data = preprocessor.transform(data)

        # Make predictions based on model type
        predictions_list = []

        if model_type == "fraud":
            preds = model.predict(data)
            probs = model.predict_fraud_probability(data)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                predictions_list.append(
                    {
                        "index": i,
                        "is_fraud": bool(pred),
                        "fraud_probability": float(prob),
                        "risk_level": _get_risk_level(prob),
                    }
                )

        elif model_type == "price":
            preds = model.predict(data)
            for i, pred in enumerate(preds):
                predictions_list.append(
                    {
                        "index": i,
                        "predicted_price": float(pred),
                        "price_range_low": float(pred * 0.85),
                        "price_range_high": float(pred * 1.15),
                    }
                )

        elif model_type == "churn":
            preds = model.predict(data)
            probs = model.predict_churn_probability(data)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                predictions_list.append(
                    {
                        "index": i,
                        "will_churn": bool(pred),
                        "churn_probability": float(prob),
                        "risk_segment": _get_risk_segment(prob),
                    }
                )

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
    """
    List all available models and their loading status.
    """
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
