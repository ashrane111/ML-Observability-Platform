"""
Data Schemas Module

Pydantic models for all datasets used in the ML Observability Platform.
Defines schemas for:
- Fraud Detection (transactions)
- Price Prediction (housing/products)
- Churn Prediction (customers)
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class TransactionType(str, Enum):
    """Types of transactions for fraud detection."""

    PURCHASE = "purchase"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    PAYMENT = "payment"


class MerchantCategory(str, Enum):
    """Merchant categories for transactions."""

    GROCERY = "grocery"
    RESTAURANT = "restaurant"
    GAS_STATION = "gas_station"
    ONLINE_SHOPPING = "online_shopping"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    OTHER = "other"


class PropertyType(str, Enum):
    """Types of properties for price prediction."""

    HOUSE = "house"
    APARTMENT = "apartment"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"


class SubscriptionPlan(str, Enum):
    """Subscription plans for churn prediction."""

    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class PaymentMethod(str, Enum):
    """Payment methods."""

    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"


# =============================================================================
# Fraud Detection Schema
# =============================================================================


class Transaction(BaseModel):
    """Schema for a financial transaction (Fraud Detection)."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp")

    # Transaction details
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    merchant_category: MerchantCategory = Field(..., description="Merchant category")

    # Location features
    latitude: float = Field(..., ge=-90, le=90, description="Transaction latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Transaction longitude")
    distance_from_home: float = Field(..., ge=0, description="Distance from user's home in km")

    # User behavior features
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_weekend: bool = Field(..., description="Whether transaction is on weekend")

    # Historical features
    avg_transaction_amount: float = Field(
        ..., ge=0, description="User's average transaction amount"
    )
    transaction_count_24h: int = Field(..., ge=0, description="Number of transactions in last 24h")
    transaction_count_7d: int = Field(
        ..., ge=0, description="Number of transactions in last 7 days"
    )

    # Device/Channel features
    is_online: bool = Field(..., description="Whether transaction is online")
    is_foreign: bool = Field(..., description="Whether transaction is in foreign country")

    # Target
    is_fraud: Optional[bool] = Field(None, description="Whether transaction is fraudulent")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_001",
                "timestamp": "2024-01-15T14:30:00Z",
                "amount": 150.00,
                "transaction_type": "purchase",
                "merchant_category": "retail",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "distance_from_home": 5.2,
                "hour_of_day": 14,
                "day_of_week": 0,
                "is_weekend": False,
                "avg_transaction_amount": 85.50,
                "transaction_count_24h": 3,
                "transaction_count_7d": 15,
                "is_online": False,
                "is_foreign": False,
                "is_fraud": False,
            }
        }


# =============================================================================
# Price Prediction Schema
# =============================================================================


class Property(BaseModel):
    """Schema for a property listing (Price Prediction)."""

    property_id: str = Field(..., description="Unique property identifier")
    listing_date: datetime = Field(..., description="Date property was listed")

    # Property characteristics
    property_type: PropertyType = Field(..., description="Type of property")
    square_feet: float = Field(..., gt=0, description="Total square footage")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=15, description="Number of bathrooms")
    year_built: int = Field(..., ge=1800, le=2030, description="Year property was built")

    # Location features
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude")
    neighborhood_score: float = Field(..., ge=0, le=10, description="Neighborhood quality score")
    school_rating: float = Field(..., ge=0, le=10, description="Nearby school rating")
    crime_rate: float = Field(..., ge=0, description="Local crime rate per 1000 residents")

    # Amenities
    has_garage: bool = Field(..., description="Whether property has garage")
    has_pool: bool = Field(..., description="Whether property has pool")
    has_garden: bool = Field(..., description="Whether property has garden")
    renovated: bool = Field(..., description="Whether property was recently renovated")

    # Market features
    days_on_market: int = Field(..., ge=0, description="Days property has been listed")
    num_price_changes: int = Field(..., ge=0, description="Number of price changes")

    # Target
    price: Optional[float] = Field(None, ge=0, description="Property price in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "property_id": "prop_001",
                "listing_date": "2024-01-15",
                "property_type": "house",
                "square_feet": 2500,
                "bedrooms": 4,
                "bathrooms": 2.5,
                "year_built": 1995,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "neighborhood_score": 7.5,
                "school_rating": 8.0,
                "crime_rate": 2.5,
                "has_garage": True,
                "has_pool": False,
                "has_garden": True,
                "renovated": True,
                "days_on_market": 30,
                "num_price_changes": 1,
                "price": 450000,
            }
        }


# =============================================================================
# Churn Prediction Schema
# =============================================================================


class Customer(BaseModel):
    """Schema for a customer record (Churn Prediction)."""

    customer_id: str = Field(..., description="Unique customer identifier")
    signup_date: datetime = Field(..., description="Date customer signed up")

    # Demographics
    age: int = Field(..., ge=18, le=100, description="Customer age")
    gender: str = Field(..., description="Customer gender")
    location: str = Field(..., description="Customer location/region")

    # Subscription info
    subscription_plan: SubscriptionPlan = Field(..., description="Current subscription plan")
    monthly_charges: float = Field(..., ge=0, description="Monthly subscription charges")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    payment_method: PaymentMethod = Field(..., description="Payment method")

    # Usage metrics
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    login_frequency: float = Field(..., ge=0, description="Average logins per week")
    feature_usage_score: float = Field(..., ge=0, le=100, description="Feature usage score")
    last_activity_days: int = Field(..., ge=0, description="Days since last activity")

    # Support interactions
    support_tickets: int = Field(..., ge=0, description="Number of support tickets")
    complaints: int = Field(..., ge=0, description="Number of complaints")

    # Engagement
    email_opt_in: bool = Field(..., description="Opted in to email marketing")
    referrals: int = Field(..., ge=0, description="Number of referrals made")
    nps_score: Optional[float] = Field(None, ge=0, le=10, description="Net Promoter Score")

    # Contract
    contract_type: str = Field(..., description="Contract type (month-to-month, annual)")
    auto_renewal: bool = Field(..., description="Whether auto-renewal is enabled")

    # Target
    churned: Optional[bool] = Field(None, description="Whether customer churned")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "cust_001",
                "signup_date": "2023-01-15",
                "age": 35,
                "gender": "female",
                "location": "northeast",
                "subscription_plan": "premium",
                "monthly_charges": 49.99,
                "total_charges": 599.88,
                "payment_method": "credit_card",
                "tenure_months": 12,
                "login_frequency": 5.5,
                "feature_usage_score": 72.5,
                "last_activity_days": 2,
                "support_tickets": 1,
                "complaints": 0,
                "email_opt_in": True,
                "referrals": 2,
                "nps_score": 8.5,
                "contract_type": "annual",
                "auto_renewal": True,
                "churned": False,
            }
        }


# =============================================================================
# Batch/Dataset Schemas
# =============================================================================


class DatasetMetadata(BaseModel):
    """Metadata for a generated dataset."""

    dataset_id: str = Field(..., description="Unique dataset identifier")
    dataset_type: str = Field(..., description="Type of dataset (fraud/price/churn)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    num_samples: int = Field(..., ge=0, description="Number of samples")

    # Split information
    split: str = Field(default="full", description="Dataset split (train/test/full)")

    # Drift information
    has_drift: bool = Field(default=False, description="Whether drift was injected")
    drift_type: Optional[str] = Field(None, description="Type of drift injected")
    drift_magnitude: Optional[float] = Field(None, description="Magnitude of drift")
    drift_features: Optional[list[str]] = Field(None, description="Features affected by drift")

    # Statistics
    target_distribution: Optional[dict] = Field(None, description="Distribution of target variable")
    feature_statistics: Optional[dict] = Field(None, description="Feature statistics")


class PredictionRequest(BaseModel):
    """Schema for a prediction request."""

    model_name: str = Field(..., description="Name of the model to use")
    features: dict = Field(..., description="Feature values for prediction")
    return_explanation: bool = Field(
        default=False, description="Whether to return SHAP explanation"
    )


class PredictionResponse(BaseModel):
    """Schema for a prediction response."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")

    prediction: float | int | bool = Field(..., description="Model prediction")
    probability: Optional[float] = Field(
        None, description="Prediction probability (for classification)"
    )
    confidence: Optional[float] = Field(None, description="Prediction confidence score")

    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional explanation
    explanation: Optional[dict] = Field(None, description="SHAP explanation if requested")
