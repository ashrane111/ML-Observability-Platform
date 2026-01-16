"""Explainability API routes for SHAP-based explanations."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explainability"])


# ============================================================================
# Request/Response Schemas
# ============================================================================


class ExplainRequest(BaseModel):
    """Request for prediction explanation."""

    model_name: str = Field(..., description="Name of the model (fraud, price, churn)")
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of top features to return")

    model_config = {"json_schema_extra": {"example": {
        "model_name": "fraud",
        "features": {
            "amount": 150.0,
            "hour": 14,
            "day_of_week": 2,
            "is_weekend": 0,
            "merchant_category": "retail",
            "customer_age": 35,
            "account_age_days": 730,
            "transaction_count_24h": 3,
            "avg_transaction_amount": 85.0,
            "distance_from_home": 5.2,
        },
        "top_k": 10,
    }}}


class FeatureContribution(BaseModel):
    """Single feature contribution to prediction."""

    feature: str
    contribution: float


class ExplanationResponse(BaseModel):
    """Response with prediction explanation."""

    model_name: str
    prediction: float
    base_value: Optional[float] = None
    shap_values: Dict[str, float] = Field(description="SHAP value for each feature")
    top_positive: List[FeatureContribution] = Field(description="Features pushing prediction up")
    top_negative: List[FeatureContribution] = Field(description="Features pushing prediction down")
    total_contribution: float


class FeatureImportanceRequest(BaseModel):
    """Request for global feature importance."""

    model_name: str = Field(..., description="Name of the model")
    method: str = Field(
        default="mean_abs",
        description="Aggregation method: mean_abs, mean, or max_abs"
    )
    top_k: int = Field(default=20, ge=1, le=100, description="Number of top features")


class FeatureImportanceResponse(BaseModel):
    """Response with global feature importance."""

    model_name: str
    method: str
    feature_importance: Dict[str, float]
    top_features: List[str]


class BatchExplainRequest(BaseModel):
    """Request for batch explanations."""

    model_name: str
    samples: List[Dict[str, Any]]
    top_k: int = Field(default=5, ge=1, le=20)


class BatchExplanationItem(BaseModel):
    """Single explanation in batch response."""

    sample_index: int
    prediction: float
    top_positive: List[FeatureContribution]
    top_negative: List[FeatureContribution]


class BatchExplainResponse(BaseModel):
    """Response with batch explanations."""

    model_name: str
    explanations: List[BatchExplanationItem]


# ============================================================================
# Dependency: Get explainer registry and model manager
# ============================================================================

# These will be injected at runtime when the router is included
_explainer_registry = None
_model_manager = None


def set_dependencies(explainer_registry, model_manager):
    """Set runtime dependencies for the router."""
    global _explainer_registry, _model_manager
    _explainer_registry = explainer_registry
    _model_manager = model_manager


def get_explainer(model_name: str):
    """Get explainer for model, creating if needed."""
    if _explainer_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability service not initialized"
        )

    explainer = _explainer_registry.get(model_name)
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No explainer available for model: {model_name}"
        )
    return explainer


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/prediction", response_model=ExplanationResponse)
async def explain_prediction(request: ExplainRequest) -> ExplanationResponse:
    """Get SHAP explanation for a single prediction.

    Returns:
    - SHAP values for each feature
    - Top positive contributors (features pushing prediction up)
    - Top negative contributors (features pushing prediction down)
    - Base value (expected model output)
    """
    import pandas as pd
    import numpy as np
    from src.api.routes.predictions import (
        prepare_fraud_features,
        prepare_price_features,
        prepare_churn_features,
    )

    explainer = get_explainer(request.model_name)

    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])

        # Encode features based on model type
        if request.model_name == "fraud":
            df_encoded = prepare_fraud_features(df).astype(np.float64)
        elif request.model_name == "price":
            df_encoded = prepare_price_features(df).astype(np.float64)
        elif request.model_name == "churn":
            df_encoded = prepare_churn_features(df).astype(np.float64)
        else:
            df_encoded = df.astype(np.float64)

        # Get explanation with encoded features
        explanation = explainer.explain_prediction(df_encoded)

        # Get feature contributions (already sorted by absolute value)
        feature_contributions = explanation.get("feature_contributions", {})
        
        # Build SHAP values dict
        shap_values = feature_contributions

        # Separate positive and negative contributors
        positive_contributors = [
            FeatureContribution(feature=k, contribution=v)
            for k, v in feature_contributions.items()
            if v > 0
        ][:request.top_k]
        
        negative_contributors = [
            FeatureContribution(feature=k, contribution=v)
            for k, v in feature_contributions.items()
            if v < 0
        ][:request.top_k]

        # Get prediction value
        pred = explanation.get("prediction", [0])
        prediction_value = pred[0] if isinstance(pred, list) else pred

        return ExplanationResponse(
            model_name=request.model_name,
            prediction=float(prediction_value),
            base_value=explanation.get("base_value"),
            shap_values=shap_values,
            top_positive=positive_contributors,
            top_negative=negative_contributors,
            total_contribution=sum(shap_values.values()),
        )

    except Exception as e:
        logger.error(f"Explanation failed for {request.model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@router.post("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(request: FeatureImportanceRequest) -> FeatureImportanceResponse:
    """Get global feature importance for a model.

    Uses mean absolute SHAP values computed on reference data.
    """
    explainer = get_explainer(request.model_name)

    try:
        importance = explainer.get_feature_importance(method=request.method)

        # Get top k features
        top_features = list(importance.keys())[:request.top_k]
        filtered_importance = {k: v for k, v in importance.items() if k in top_features}

        return FeatureImportanceResponse(
            model_name=request.model_name,
            method=request.method,
            feature_importance=filtered_importance,
            top_features=top_features,
        )

    except Exception as e:
        logger.error(f"Feature importance failed for {request.model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature importance calculation failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchExplainResponse)
async def explain_batch(request: BatchExplainRequest) -> BatchExplainResponse:
    """Get explanations for multiple predictions.

    Limited to 100 samples per request for performance.
    """
    import pandas as pd

    if len(request.samples) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 100 samples"
        )

    explainer = get_explainer(request.model_name)

    try:
        df = pd.DataFrame(request.samples)
        contributions = explainer.get_feature_contributions(df, top_k=request.top_k)

        explanations = [
            BatchExplanationItem(
                sample_index=c["sample_index"],
                prediction=c["prediction"],
                top_positive=[FeatureContribution(**f) for f in c["positive_contributors"]],
                top_negative=[FeatureContribution(**f) for f in c["negative_contributors"]],
            )
            for c in contributions
        ]

        return BatchExplainResponse(
            model_name=request.model_name,
            explanations=explanations,
        )

    except Exception as e:
        logger.error(f"Batch explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch explanation failed: {str(e)}"
        )


@router.get("/models")
async def list_explainable_models() -> Dict[str, Any]:
    """List all models with explainability enabled."""
    if _explainer_registry is None:
        return {"models": [], "status": "not_initialized"}

    models = _explainer_registry.list_models()
    return {
        "models": models,
        "count": len(models),
        "status": "ready",
    }


@router.get("/report/{model_name}")
async def get_explanation_report(
    model_name: str,
    sample_data: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive explanation report for a model.

    If sample_data is provided (JSON string), explains that sample.
    Otherwise, uses a sample from reference data.
    """
    import json

    import pandas as pd

    explainer = get_explainer(model_name)

    try:
        if sample_data:
            data = json.loads(sample_data)
            df = pd.DataFrame([data])
        elif explainer._background_data is not None:
            # Use first sample from background data
            df = explainer._background_data.iloc[[0]]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sample data provided and no reference data available"
            )

        report = explainer.generate_explanation_report(df, sample_idx=0)
        report["model_name"] = model_name

        return report

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in sample_data"
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )