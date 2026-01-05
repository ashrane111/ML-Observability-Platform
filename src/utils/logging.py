"""
Logging Configuration Module

Provides structured logging with loguru for the entire application.
Supports JSON output, log rotation, and different log levels per environment.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.config import settings


def serialize_record(record: dict[str, Any]) -> str:
    """Serialize log record to JSON format."""
    import json
    from datetime import datetime

    subset = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add extra fields if present
    if record.get("extra"):
        subset["extra"] = record["extra"]

    # Add exception info if present
    if record["exception"]:
        subset["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
            "traceback": record["exception"].traceback is not None,
        }

    return json.dumps(subset, default=str)


def json_sink(message: Any) -> None:
    """Custom sink for JSON formatted logs."""
    record = message.record
    serialized = serialize_record(record)
    print(serialized, file=sys.stdout, flush=True)


def setup_logging() -> None:
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()

    # Determine log format based on environment
    if settings.is_production:
        # JSON format for production (easier to parse in log aggregators)
        logger.add(
            json_sink,
            level=settings.log_level,
            serialize=False,
        )
    else:
        # Human-readable format for development
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=log_format,
            level=settings.log_level,
            colorize=True,
        )

    # Add file logging with rotation
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Main log file with rotation
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="gz",  # Compress old logs
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
    )

    # Error log file (errors only)
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="90 days",
        compression="gz",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
    )

    logger.info(
        f"Logging configured: level={settings.log_level}, env={settings.app_env}"
    )


def get_logger(name: str = __name__) -> "logger":
    """
    Get a logger instance with context.

    Args:
        name: Logger name (usually module name)

    Returns:
        Logger instance bound with the given name
    """
    return logger.bind(logger_name=name)


# Initialize logging on module import
setup_logging()
