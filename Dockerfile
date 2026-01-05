# =============================================================================
# ML Observability Platform - Dockerfile
# =============================================================================
# Multi-stage build for API and Dashboard services
# =============================================================================

# -----------------------------------------------------------------------------
# Base Stage - Common dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# -----------------------------------------------------------------------------
# Dependencies Stage - Install Python packages
# -----------------------------------------------------------------------------
FROM base as dependencies

# Copy only dependency files first (for caching)
COPY pyproject.toml ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# -----------------------------------------------------------------------------
# API Stage - FastAPI application
# -----------------------------------------------------------------------------
FROM dependencies as api

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create directories for prometheus multiprocess mode
RUN mkdir -p /tmp/prometheus && \
    chown -R appuser:appgroup /tmp/prometheus

# Create data directories with proper permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/data/reference /app/models && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Dashboard Stage - Streamlit application
# -----------------------------------------------------------------------------
FROM dependencies as dashboard

# Copy dashboard code
COPY dashboard/ ./dashboard/
COPY src/ ./src/

# Create directories with proper permissions
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# -----------------------------------------------------------------------------
# Development Stage - With dev dependencies
# -----------------------------------------------------------------------------
FROM dependencies as development

# Install dev dependencies
RUN pip install -e ".[dev,notebooks]"

# Copy all code
COPY . .

# Create directories with proper permissions
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Default command (can be overridden)
CMD ["bash"]
