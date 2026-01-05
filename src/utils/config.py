"""
Configuration Management Module

Handles all application configuration using Pydantic Settings.
Loads from environment variables and .env files.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application Settings
    # =========================================================================
    app_name: str = Field(default="ml-observability-platform")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    app_debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # =========================================================================
    # API Settings
    # =========================================================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)
    api_workers: int = Field(default=1)

    # =========================================================================
    # Database Settings
    # =========================================================================
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="mlobs")
    postgres_password: str = Field(default="mlobs_secret_password")
    postgres_db: str = Field(default="mlobs")
    database_url: str | None = Field(default=None)

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_url(cls, v: str | None, info) -> str:
        if v:
            return v
        data = info.data
        return (
            f"postgresql://{data.get('postgres_user', 'mlobs')}:"
            f"{data.get('postgres_password', 'mlobs_secret_password')}@"
            f"{data.get('postgres_host', 'localhost')}:"
            f"{data.get('postgres_port', 5432)}/"
            f"{data.get('postgres_db', 'mlobs')}"
        )

    # =========================================================================
    # Redis Settings
    # =========================================================================
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str | None = Field(default=None)
    redis_url: str | None = Field(default=None)

    @field_validator("redis_url", mode="before")
    @classmethod
    def assemble_redis_url(cls, v: str | None, info) -> str:
        if v:
            return v
        data = info.data
        password = data.get("redis_password")
        if password:
            return f"redis://:{password}@{data.get('redis_host', 'localhost')}:{data.get('redis_port', 6379)}"
        return f"redis://{data.get('redis_host', 'localhost')}:{data.get('redis_port', 6379)}"

    # =========================================================================
    # MLflow Settings
    # =========================================================================
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="ml-observability")

    # =========================================================================
    # OpenTelemetry Settings
    # =========================================================================
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="ml-observability-platform")
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)

    # =========================================================================
    # Prometheus Settings
    # =========================================================================
    prometheus_multiproc_dir: str = Field(default="/tmp/prometheus")

    # =========================================================================
    # Prefect Settings
    # =========================================================================
    prefect_api_url: str = Field(default="http://localhost:4200/api")

    # =========================================================================
    # Monitoring Thresholds
    # =========================================================================
    model_drift_threshold_psi: float = Field(default=0.2)
    model_drift_threshold_critical: float = Field(default=0.3)
    data_quality_threshold: float = Field(default=0.8)
    performance_degradation_threshold: float = Field(default=0.1)

    # =========================================================================
    # Feature Flags
    # =========================================================================
    enable_drift_detection: bool = Field(default=True)
    enable_explainability: bool = Field(default=True)
    enable_alerting: bool = Field(default=True)

    # =========================================================================
    # Monitoring Intervals (in minutes)
    # =========================================================================
    drift_check_interval_minutes: int = Field(default=60)
    data_quality_check_interval_minutes: int = Field(default=1440)  # Daily

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
