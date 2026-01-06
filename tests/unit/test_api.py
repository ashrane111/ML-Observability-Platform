"""
Unit Tests for API Module

Tests for API endpoints using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

# from src.api.routes.health import set_model_status
# from src.api.routes.predictions import set_model
# from src.api.schemas import ModelType


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data

    def test_liveness_probe(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe(self, client):
        """Test readiness probe."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data

    def test_version_info(self, client):
        """Test version info endpoint."""
        response = client.get("/health/version")

        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "service" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ML Observability Platform"
        assert "version" in data
        assert "docs" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_fraud_prediction_no_model(self, client):
        """Test fraud prediction when model not loaded."""
        request_data = {
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

        response = client.post("/predict/fraud", json=request_data)

        # Should return 503 when model not loaded
        assert response.status_code == 503

    def test_price_prediction_no_model(self, client):
        """Test price prediction when model not loaded."""
        request_data = {
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

        response = client.post("/predict/price", json=request_data)

        # Should return 503 when model not loaded
        assert response.status_code == 503

    def test_churn_prediction_no_model(self, client):
        """Test churn prediction when model not loaded."""
        request_data = {
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

        response = client.post("/predict/churn", json=request_data)

        # Should return 503 when model not loaded
        assert response.status_code == 503

    def test_list_models(self, client):
        """Test list models endpoint."""
        response = client.get("/predict/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "fraud" in data["models"]
        assert "price" in data["models"]
        assert "churn" in data["models"]

    def test_fraud_prediction_invalid_data(self, client):
        """Test fraud prediction with invalid data."""
        request_data = {
            "amount": -100,  # Invalid: negative amount
        }

        response = client.post("/predict/fraud", json=request_data)

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_batch_prediction_no_model(self, client):
        """Test batch prediction when model not loaded."""
        request_data = {
            "model_type": "fraud",
            "instances": [
                {
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
            ],
        }

        response = client.post("/predict/batch", json=request_data)

        # Should return 503 when model not loaded
        assert response.status_code == 503


class TestMonitoringEndpoints:
    """Tests for monitoring endpoints."""

    def test_drift_status_no_detector(self, client):
        """Test drift status when detector not initialized."""
        response = client.get("/monitoring/drift/status/fraud")

        # Should return 503 when detector not initialized
        assert response.status_code == 503

    def test_list_alerts(self, client):
        """Test list alerts endpoint."""
        response = client.get("/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total" in data

    def test_alert_summary(self, client):
        """Test alert summary endpoint."""
        response = client.get("/monitoring/alerts/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_active" in data
        assert "by_severity" in data
        assert "by_type" in data

    def test_get_alert_not_found(self, client):
        """Test get alert with non-existent ID."""
        response = client.get("/monitoring/alerts/non-existent-id")

        assert response.status_code == 404

    def test_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/monitoring/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_drift_check_no_detector(self, client):
        """Test drift check when detector not initialized."""
        request_data = {
            "model_type": "fraud",
            "data": [{"feature1": 1.0, "feature2": 2.0}],
            "dataset_name": "test",
        }

        response = client.post("/monitoring/drift/check", json=request_data)

        # Should return 503 when detector not initialized
        assert response.status_code == 503

    def test_data_quality_check_no_detector(self, client):
        """Test data quality check when detector not initialized."""
        request_data = {
            "model_type": "fraud",
            "data": [{"feature1": 1.0, "feature2": 2.0}],
            "dataset_name": "test",
        }

        response = client.post("/monitoring/quality/check", json=request_data)

        # Should return 503 when detector not initialized
        assert response.status_code == 503


class TestAlertManagement:
    """Tests for alert management endpoints."""

    def test_acknowledge_alert_not_found(self, client):
        """Test acknowledging non-existent alert."""
        response = client.post(
            "/monitoring/alerts/non-existent-id/acknowledge",
            json={"acknowledged_by": "test_user"},
        )

        assert response.status_code == 404

    def test_resolve_alert_not_found(self, client):
        """Test resolving non-existent alert."""
        response = client.post(
            "/monitoring/alerts/non-existent-id/resolve",
            json={"resolved_by": "test_user", "resolution_notes": "Fixed"},
        )

        assert response.status_code == 404


class TestSchemaValidation:
    """Tests for request schema validation."""

    def test_fraud_request_missing_fields(self, client):
        """Test fraud prediction with missing required fields."""
        response = client.post("/predict/fraud", json={})

        assert response.status_code == 422

    def test_price_request_invalid_coordinates(self, client):
        """Test price prediction with invalid coordinates."""
        request_data = {
            "property_type": "house",
            "square_feet": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1995,
            "latitude": 200,  # Invalid: > 90
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

        response = client.post("/predict/price", json=request_data)

        assert response.status_code == 422

    def test_churn_request_invalid_age(self, client):
        """Test churn prediction with invalid age."""
        request_data = {
            "age": 5,  # Invalid: < 18
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

        response = client.post("/predict/churn", json=request_data)

        assert response.status_code == 422

    def test_batch_request_empty_instances(self, client):
        """Test batch prediction with empty instances."""
        request_data = {
            "model_type": "fraud",
            "instances": [],
        }

        response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 422

    def test_drift_check_empty_data(self, client):
        """Test drift check with empty data."""
        request_data = {
            "model_type": "fraud",
            "data": [],
        }

        response = client.post("/monitoring/drift/check", json=request_data)

        assert response.status_code == 422
