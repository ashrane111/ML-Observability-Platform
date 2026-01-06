# =============================================================================
# ML Observability Platform - Makefile
# =============================================================================

.PHONY: help install dev test lint format clean docker-build docker-up docker-down \
        generate-data train-models run-api setup-reference

# Default target
help:
	@echo "ML Observability Platform - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         Install production dependencies"
	@echo "  make dev             Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make lint            Run linters (flake8, mypy)"
	@echo "  make format          Format code (black, isort)"
	@echo "  make clean           Remove cache and build files"
	@echo ""
	@echo "Data & Models:"
	@echo "  make generate-data   Generate synthetic datasets"
	@echo "  make train-models    Train all ML models"
	@echo "  make setup-reference Setup reference data for drift detection"
	@echo ""
	@echo "API:"
	@echo "  make run-api         Run API server locally"
	@echo "  make run-api-dev     Run API server with hot reload"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker images"
	@echo "  make docker-up       Start all services"
	@echo "  make docker-down     Stop all services"
	@echo "  make docker-logs     View service logs"
	@echo "  make docker-clean    Remove containers and volumes"
	@echo ""

# =============================================================================
# Setup & Installation
# =============================================================================

install:
	uv sync --frozen --no-dev

dev:
	uv sync --frozen

# =============================================================================
# Development
# =============================================================================

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true

# =============================================================================
# Data & Models
# =============================================================================

generate-data:
	@echo "Generating synthetic datasets..."
	python scripts/generate_data.py --output-dir data --seed 42
	@echo "Data generation complete!"

train-models:
	@echo "Training ML models..."
	python scripts/train_models.py --data-dir data --output-dir models --experiment ml-observability
	@echo "Model training complete!"

setup-reference:
	@echo "Setting up reference data for drift detection..."
	@mkdir -p data/reference
	@if [ -f data/fraud_reference.parquet ]; then \
		cp data/fraud_reference.parquet data/reference/; \
	fi
	@if [ -f data/price_reference.parquet ]; then \
		cp data/price_reference.parquet data/reference/; \
	fi
	@if [ -f data/churn_reference.parquet ]; then \
		cp data/churn_reference.parquet data/reference/; \
	fi
	@echo "Reference data setup complete!"

# =============================================================================
# API
# =============================================================================

run-api:
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

run-api-dev:
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo ""
	@echo "Services started!"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo ""

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --rmi local
	docker system prune -f

docker-restart: docker-down docker-up

# =============================================================================
# Full Setup
# =============================================================================

setup: dev generate-data train-models setup-reference
	@echo ""
	@echo "Full setup complete!"
	@echo "Run 'make run-api-dev' to start the API server"
	@echo "Or run 'make docker-up' to start with Docker"
	@echo ""

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci-test:
	pytest tests/ -v --cov=src --cov-report=xml

ci-lint:
	flake8 src/ tests/ scripts/ --format=default
	mypy src/ --ignore-missing-imports --no-error-summary
