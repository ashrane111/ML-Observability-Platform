"""
FastAPI Application

Main application entry point for the ML Observability Platform API.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import health, monitoring, predictions
from src.api.routes.health import set_model_status
from src.api.routes.monitoring import set_drift_detector
from src.api.routes.predictions import set_model
from src.api.schemas import ErrorResponse
from src.models.churn_predictor import ChurnPredictor
from src.models.fraud_detector import FraudDetector

# from src.models.preprocessing import FeaturePreprocessor
from src.models.price_predictor import PricePredictor
from src.monitoring.drift_detector import DriftDetector

# Configuration
MODEL_DIR = Path("models")
DATA_DIR = Path("data")


def load_models() -> dict[str, Any]:
    """Load all trained models."""
    models = {}

    # Load Fraud Detector
    fraud_model_path = MODEL_DIR / "fraud_detector"
    if fraud_model_path.exists():
        try:
            model_files = list(fraud_model_path.glob("*.pkl"))
            if model_files:
                models["fraud"] = FraudDetector.load(model_files[0])
                logger.info(f"Loaded fraud detector: {model_files[0]}")
        except Exception as e:
            logger.error(f"Failed to load fraud detector: {e}")

    # Load Price Predictor
    price_model_path = MODEL_DIR / "price_predictor"
    if price_model_path.exists():
        try:
            model_files = list(price_model_path.glob("*.pkl"))
            if model_files:
                models["price"] = PricePredictor.load(model_files[0])
                logger.info(f"Loaded price predictor: {model_files[0]}")
        except Exception as e:
            logger.error(f"Failed to load price predictor: {e}")

    # Load Churn Predictor
    churn_model_path = MODEL_DIR / "churn_predictor"
    if churn_model_path.exists():
        try:
            model_files = list(churn_model_path.glob("*.pkl"))
            if model_files:
                models["churn"] = ChurnPredictor.load(model_files[0])
                logger.info(f"Loaded churn predictor: {model_files[0]}")
        except Exception as e:
            logger.error(f"Failed to load churn predictor: {e}")

    return models


def load_reference_data() -> dict[str, pd.DataFrame]:
    """Load reference datasets for drift detection."""
    reference_data: dict[str, pd.DataFrame] = {}

    reference_dir = DATA_DIR / "reference"
    if not reference_dir.exists():
        logger.warning(f"Reference data directory not found: {reference_dir}")
        return reference_data

    # Load fraud reference
    fraud_ref = reference_dir / "fraud_reference.parquet"
    if fraud_ref.exists():
        try:
            reference_data["fraud"] = pd.read_parquet(fraud_ref)
            logger.info(f"Loaded fraud reference: {len(reference_data['fraud'])} samples")
        except Exception as e:
            logger.error(f"Failed to load fraud reference: {e}")

    # Load price reference
    price_ref = reference_dir / "price_reference.parquet"
    if price_ref.exists():
        try:
            reference_data["price"] = pd.read_parquet(price_ref)
            logger.info(f"Loaded price reference: {len(reference_data['price'])} samples")
        except Exception as e:
            logger.error(f"Failed to load price reference: {e}")

    # Load churn reference
    churn_ref = reference_dir / "churn_reference.parquet"
    if churn_ref.exists():
        try:
            reference_data["churn"] = pd.read_parquet(churn_ref)
            logger.info(f"Loaded churn reference: {len(reference_data['churn'])} samples")
        except Exception as e:
            logger.error(f"Failed to load churn reference: {e}")

    return reference_data


def setup_drift_detectors(reference_data: dict[str, pd.DataFrame]) -> dict[str, DriftDetector]:
    """Setup drift detectors with reference data."""
    detectors = {}

    target_columns = {
        "fraud": "is_fraud",
        "price": "price",
        "churn": "churned",
    }

    for model_type, ref_data in reference_data.items():
        try:
            detector = DriftDetector(
                psi_threshold_warning=0.1,
                psi_threshold_critical=0.2,
                drift_share_threshold=0.3,
            )
            detector.set_reference_data(
                ref_data,
                target_column=target_columns.get(model_type),
            )
            detectors[model_type] = detector
            logger.info(f"Drift detector configured for {model_type}")
        except Exception as e:
            logger.error(f"Failed to setup drift detector for {model_type}: {e}")

    return detectors


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting ML Observability Platform API...")

    # Load models
    models = load_models()
    for model_type, model in models.items():
        set_model(model_type, model)
        set_model_status(
            f"{model_type}_detector" if model_type == "fraud" else f"{model_type}_predictor", True
        )

    # Load reference data and setup drift detectors
    reference_data = load_reference_data()
    detectors = setup_drift_detectors(reference_data)
    for model_type, detector in detectors.items():
        set_drift_detector(model_type, detector)

    logger.info(f"Loaded {len(models)} models, {len(detectors)} drift detectors")
    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down ML Observability Platform API...")


# Create FastAPI application
app = FastAPI(
    title="ML Observability Platform",
    description="""
    A comprehensive ML observability platform providing:

    - **Predictions**: Real-time predictions for fraud detection, \
          price prediction, and churn prediction
    - **Drift Detection**: Monitor data drift using PSI and statistical tests
    - **Data Quality**: Check data quality metrics
    - **Alerts**: Manage monitoring alerts
    - **Metrics**: Prometheus-compatible metrics endpoint

    ## Models

    - **Fraud Detector**: XGBoost classifier for transaction fraud detection
    - **Price Predictor**: LightGBM regressor for property price prediction
    - **Churn Predictor**: Random Forest classifier for customer churn prediction
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Middleware
# =============================================================================


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log all requests."""
    logger.debug(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"{request.method} {request.url.path} - {response.status_code}")
    return response


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            detail={"exception": str(exc)},
        ).model_dump(mode="json"),
    )


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(monitoring.router)


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "ML Observability Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Run with Uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",  # nosec B104
        port=8000,
        reload=True,
        log_level="info",
    )
