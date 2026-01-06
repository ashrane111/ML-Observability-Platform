"""
API Schemas Module

Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class ModelType(str, Enum):
    """Available model types."""

    FRAUD = "fraud"
    PRICE = "price"
    CHURN = "churn"


class DriftStatusEnum(str, Enum):
    """Drift detection status."""

    NO_DRIFT = "no_drift"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertSeverityEnum(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# Health Check Schemas
# =============================================================================


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")
    models_loaded: dict[str, bool] = Field(default_factory=dict, description="Model loading status")


class ReadinessStatus(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether service is ready")
    checks: dict[str, bool] = Field(default_factory=dict, description="Individual check results")


# =============================================================================
# Prediction Schemas
# =============================================================================


class FraudPredictionRequest(BaseModel):
    """Request schema for fraud prediction."""

    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_type: str = Field(..., description="Type of transaction")
    merchant_category: str = Field(..., description="Merchant category")
    latitude: float = Field(..., ge=-90, le=90, description="Transaction latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Transaction longitude")
    distance_from_home: float = Field(..., ge=0, description="Distance from home")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week")
    is_weekend: bool = Field(..., description="Whether it's a weekend")
    avg_transaction_amount: float = Field(..., gt=0, description="Average transaction amount")
    transaction_count_24h: int = Field(..., ge=0, description="Transactions in last 24h")
    transaction_count_7d: int = Field(..., ge=0, description="Transactions in last 7 days")
    is_online: bool = Field(..., description="Whether transaction is online")
    is_foreign: bool = Field(..., description="Whether transaction is foreign")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 150.00,
                "transaction_type": "purchase",
                "merchant_category": "retail",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "distance_from_home": 5.2,
                "hour_of_day": 14,
                "day_of_week": 2,
                "is_weekend": False,
                "avg_transaction_amount": 125.00,
                "transaction_count_24h": 3,
                "transaction_count_7d": 15,
                "is_online": False,
                "is_foreign": False,
            }
        }
    }


class FraudPredictionResponse(BaseModel):
    """Response schema for fraud prediction."""

    is_fraud: bool = Field(..., description="Fraud prediction")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")
    risk_level: str = Field(..., description="Risk level (low/medium/high/critical)")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction ID")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class PricePredictionRequest(BaseModel):
    """Request schema for price prediction."""

    property_type: str = Field(..., description="Type of property")
    square_feet: float = Field(..., gt=0, description="Square footage")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, le=15, description="Number of bathrooms")
    year_built: int = Field(..., ge=1800, le=2030, description="Year built")
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude")
    neighborhood_score: float = Field(..., ge=0, le=10, description="Neighborhood score")
    school_rating: float = Field(..., ge=0, le=10, description="School rating")
    crime_rate: float = Field(..., ge=0, description="Crime rate")
    has_garage: bool = Field(..., description="Has garage")
    has_pool: bool = Field(..., description="Has pool")
    has_garden: bool = Field(..., description="Has garden")
    renovated: bool = Field(..., description="Recently renovated")
    days_on_market: int = Field(..., ge=0, description="Days on market")
    num_price_changes: int = Field(..., ge=0, description="Number of price changes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "property_type": "house",
                "square_feet": 2000,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1995,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "neighborhood_score": 7.5,
                "school_rating": 8.0,
                "crime_rate": 2.5,
                "has_garage": True,
                "has_pool": False,
                "has_garden": True,
                "renovated": False,
                "days_on_market": 30,
                "num_price_changes": 1,
            }
        }
    }


class PricePredictionResponse(BaseModel):
    """Response schema for price prediction."""

    predicted_price: float = Field(..., description="Predicted price")
    price_range_low: float = Field(..., description="Lower bound of price range")
    price_range_high: float = Field(..., description="Upper bound of price range")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction ID")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class ChurnPredictionRequest(BaseModel):
    """Request schema for churn prediction."""

    age: int = Field(..., ge=18, le=120, description="Customer age")
    gender: str = Field(..., description="Customer gender")
    location: str = Field(..., description="Customer location")
    subscription_plan: str = Field(..., description="Subscription plan type")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    payment_method: str = Field(..., description="Payment method")
    tenure_months: int = Field(..., ge=0, description="Tenure in months")
    login_frequency: float = Field(..., ge=0, description="Login frequency")
    feature_usage_score: float = Field(..., ge=0, le=100, description="Feature usage score")
    last_activity_days: int = Field(..., ge=0, description="Days since last activity")
    support_tickets: int = Field(..., ge=0, description="Number of support tickets")
    complaints: int = Field(..., ge=0, description="Number of complaints")
    email_opt_in: bool = Field(..., description="Email opt-in status")
    referrals: int = Field(..., ge=0, description="Number of referrals")
    nps_score: int = Field(..., ge=0, le=10, description="NPS score")
    contract_type: str = Field(..., description="Contract type")
    auto_renewal: bool = Field(..., description="Auto-renewal enabled")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "gender": "female",
                "location": "urban",
                "subscription_plan": "premium",
                "monthly_charges": 79.99,
                "total_charges": 959.88,
                "payment_method": "credit_card",
                "tenure_months": 12,
                "login_frequency": 5.2,
                "feature_usage_score": 65.0,
                "last_activity_days": 3,
                "support_tickets": 1,
                "complaints": 0,
                "email_opt_in": True,
                "referrals": 2,
                "nps_score": 8,
                "contract_type": "annual",
                "auto_renewal": True,
            }
        }
    }


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""

    will_churn: bool = Field(..., description="Churn prediction")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_segment: str = Field(..., description="Risk segment")
    retention_priority: int = Field(..., ge=1, le=5, description="Retention priority (1-5)")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction ID")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    model_type: ModelType = Field(..., description="Type of model to use")
    instances: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=1000, description="List of instances to predict"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[dict[str, Any]] = Field(..., description="List of predictions")
    model_type: ModelType = Field(..., description="Model type used")
    model_version: str = Field(..., description="Model version used")
    total_instances: int = Field(..., description="Total instances processed")
    total_latency_ms: float = Field(..., description="Total processing time")
    avg_latency_ms: float = Field(..., description="Average latency per instance")


# =============================================================================
# Drift Detection Schemas
# =============================================================================


class DriftCheckRequest(BaseModel):
    """Request schema for drift check."""

    model_type: ModelType = Field(..., description="Type of model to check")
    data: list[dict[str, Any]] = Field(
        ..., min_length=1, description="Current data to check for drift"
    )
    dataset_name: Optional[str] = Field(default="current", description="Name for this dataset")


class FeatureDriftInfo(BaseModel):
    """Information about drift for a single feature."""

    feature_name: str = Field(..., description="Feature name")
    drift_score: float = Field(..., description="Drift score (PSI/JS divergence)")
    is_drifted: bool = Field(..., description="Whether drift is detected")


class DriftCheckResponse(BaseModel):
    """Response schema for drift check."""

    drift_status: DriftStatusEnum = Field(..., description="Overall drift status")
    dataset_drift_detected: bool = Field(..., description="Whether dataset drift detected")
    drift_share: float = Field(..., ge=0, le=1, description="Share of drifted features")
    drifted_features_count: int = Field(..., description="Number of drifted features")
    total_features: int = Field(..., description="Total number of features")
    feature_drift: list[FeatureDriftInfo] = Field(..., description="Per-feature drift information")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_type: ModelType = Field(..., description="Model type checked")


class DataQualityRequest(BaseModel):
    """Request schema for data quality check."""

    model_type: ModelType = Field(..., description="Type of model")
    data: list[dict[str, Any]] = Field(..., min_length=1, description="Data to check quality")
    dataset_name: Optional[str] = Field(default="current", description="Name for this dataset")


class DataQualityResponse(BaseModel):
    """Response schema for data quality check."""

    dataset_name: str = Field(..., description="Dataset name")
    total_rows: int = Field(..., description="Total number of rows")
    missing_values_share: float = Field(..., description="Share of missing values")
    columns_with_missing: list[str] = Field(..., description="Columns with missing values")
    duplicate_rows: int = Field(..., description="Number of duplicate rows")
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Alert Schemas
# =============================================================================


class AlertResponse(BaseModel):
    """Response schema for an alert."""

    alert_id: str = Field(..., description="Unique alert ID")
    alert_type: str = Field(..., description="Type of alert")
    severity: AlertSeverityEnum = Field(..., description="Alert severity")
    status: str = Field(..., description="Alert status")
    model_name: str = Field(..., description="Model name")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    created_at: datetime = Field(..., description="Creation timestamp")
    metric_name: Optional[str] = Field(None, description="Metric name")
    metric_value: Optional[float] = Field(None, description="Metric value")


class AlertListResponse(BaseModel):
    """Response schema for list of alerts."""

    alerts: list[AlertResponse] = Field(..., description="List of alerts")
    total: int = Field(..., description="Total number of alerts")


class AlertAcknowledgeRequest(BaseModel):
    """Request schema for acknowledging an alert."""

    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")


class AlertResolveRequest(BaseModel):
    """Request schema for resolving an alert."""

    resolved_by: Optional[str] = Field(None, description="Who resolved")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")


class AlertSummaryResponse(BaseModel):
    """Response schema for alert summary."""

    total_active: int = Field(..., description="Total active alerts")
    by_severity: dict[str, int] = Field(..., description="Alerts by severity")
    by_type: dict[str, int] = Field(..., description="Alerts by type")
    oldest_alert: Optional[datetime] = Field(None, description="Oldest alert timestamp")
    newest_alert: Optional[datetime] = Field(None, description="Newest alert timestamp")


# =============================================================================
# Metrics Schemas
# =============================================================================


class ModelMetricsResponse(BaseModel):
    """Response schema for model metrics."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    metrics: dict[str, float] = Field(..., description="Performance metrics")
    last_updated: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[dict[str, Any]] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.now)
