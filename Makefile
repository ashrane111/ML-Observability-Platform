# =============================================================================
# ML Observability Platform - Makefile
# =============================================================================
# Common commands for development, testing, and deployment
# =============================================================================

.PHONY: help install install-dev setup env up down restart logs \
        build clean test test-unit test-integration test-e2e coverage \
        lint format type-check pre-commit train generate-data simulate \
        demo docs api dashboard prefect shell db-migrate db-upgrade

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "$(BLUE)ML Observability Platform$(NC)"
	@echo "=========================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make setup          Complete project setup (install + env + build)"
	@echo "  make env            Create .env file from .env.example"
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@echo "  make up             Start all services"
	@echo "  make down           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make logs           View logs (all services)"
	@echo "  make logs-api       View API service logs"
	@echo "  make build          Build Docker images"
	@echo "  make clean          Remove containers, volumes, and images"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  make api            Run API locally (without Docker)"
	@echo "  make dashboard      Run Streamlit dashboard locally"
	@echo "  make prefect        Run Prefect server locally"
	@echo "  make shell          Open shell in API container"
	@echo ""
	@echo "$(GREEN)ML Commands:$(NC)"
	@echo "  make train          Train all models"
	@echo "  make generate-data  Generate synthetic data"
	@echo "  make simulate       Run traffic simulation"
	@echo "  make demo           Run demo script (drift injection)"
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-e2e       Run end-to-end tests"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Code Quality Commands:$(NC)"
	@echo "  make lint           Run linters (flake8)"
	@echo "  make format         Format code (black + isort)"
	@echo "  make type-check     Run type checker (mypy)"
	@echo "  make pre-commit     Run all pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Database Commands:$(NC)"
	@echo "  make db-migrate     Create new migration"
	@echo "  make db-upgrade     Apply migrations"
	@echo ""
	@echo "$(GREEN)Documentation Commands:$(NC)"
	@echo "  make docs           Generate documentation"
	@echo ""

# =============================================================================
# Setup Commands
# =============================================================================
install:
	@echo "$(BLUE)Installing production dependencies with uv...$(NC)"
	uv pip install -e .

install-dev:
	@echo "$(BLUE)Installing development dependencies with uv...$(NC)"
	uv pip install -e ".[dev,notebooks]"
	pre-commit install

setup: env install-dev build
	@echo "$(GREEN)Setup complete!$(NC)"
	@echo "Run 'make up' to start all services"

env:
	@if [ ! -f .env ]; then \
		echo "$(BLUE)Creating .env file from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN).env file created. Please review and update values.$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists. Skipping.$(NC)"; \
	fi

# =============================================================================
# Docker Commands
# =============================================================================
up:
	@echo "$(BLUE)Starting all services...$(NC)"
	docker compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo ""
	@echo "Access points:"
	@echo "  - API:          http://localhost:8000"
	@echo "  - API Docs:     http://localhost:8000/docs"
	@echo "  - Dashboard:    http://localhost:8501"
	@echo "  - Grafana:      http://localhost:3000"
	@echo "  - Prometheus:   http://localhost:9090"
	@echo "  - Jaeger:       http://localhost:16686"
	@echo "  - MLflow:       http://localhost:5000"
	@echo "  - Prefect:      http://localhost:4200"

down:
	@echo "$(BLUE)Stopping all services...$(NC)"
	docker compose down

restart:
	@echo "$(BLUE)Restarting all services...$(NC)"
	docker compose restart

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-dashboard:
	docker compose logs -f dashboard

build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker compose build

clean:
	@echo "$(RED)Removing containers, volumes, and images...$(NC)"
	docker compose down -v --rmi local
	rm -rf prometheus_data grafana_data postgres_data redis_data minio_data mlruns mlartifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

# =============================================================================
# Development Commands
# =============================================================================
api:
	@echo "$(BLUE)Starting API server locally...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	@echo "$(BLUE)Starting Streamlit dashboard...$(NC)"
	streamlit run dashboard/app.py --server.port 8501

prefect:
	@echo "$(BLUE)Starting Prefect server...$(NC)"
	prefect server start

shell:
	docker compose exec api /bin/bash

# =============================================================================
# ML Commands
# =============================================================================
train:
	@echo "$(BLUE)Training all models...$(NC)"
	python scripts/train_models.py

generate-data:
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	python scripts/generate_data.py

simulate:
	@echo "$(BLUE)Running traffic simulation...$(NC)"
	python scripts/simulate_traffic.py

demo:
	@echo "$(BLUE)Running demo script...$(NC)"
	python scripts/demo_drift.py

# =============================================================================
# Testing Commands
# =============================================================================
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest

test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit -v

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration -v

test-e2e:
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	pytest tests/e2e -v

coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated at htmlcov/index.html$(NC)"

# =============================================================================
# Code Quality Commands
# =============================================================================
lint:
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 src dashboard tests

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	black src dashboard tests
	isort src dashboard tests

type-check:
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy src

pre-commit:
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# =============================================================================
# Database Commands
# =============================================================================
db-migrate:
	@echo "$(BLUE)Creating new migration...$(NC)"
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-upgrade:
	@echo "$(BLUE)Applying migrations...$(NC)"
	alembic upgrade head

# =============================================================================
# Documentation Commands
# =============================================================================
docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation generation not yet implemented"

# =============================================================================
# Utility Commands
# =============================================================================
check-env:
	@echo "$(BLUE)Checking environment...$(NC)"
	@python --version
	@docker --version
	@docker compose version
	@echo "$(GREEN)Environment check complete!$(NC)"
