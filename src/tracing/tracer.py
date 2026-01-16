"""OpenTelemetry distributed tracing for ML Observability Platform.

Provides:
- Automatic trace context propagation
- Span creation for predictions and model operations
- Integration with Jaeger for trace visualization
- Custom attributes for ML-specific metadata
"""

import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    service_name: str = "ml-observability-platform"
    jaeger_endpoint: str = "http://localhost:4317"  # OTLP endpoint
    jaeger_http_endpoint: str = "http://localhost:14268/api/traces"  # HTTP Thrift
    exporter_type: str = "otlp"  # "otlp", "jaeger", or "console"
    sample_rate: float = 1.0  # 1.0 = sample everything
    enabled: bool = True
    environment: str = "development"

    # Additional resource attributes
    extra_attributes: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create config from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "ml-observability-platform"),
            jaeger_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            jaeger_http_endpoint=os.getenv(
                "JAEGER_HTTP_ENDPOINT", "http://localhost:14268/api/traces"
            ),
            exporter_type=os.getenv("OTEL_EXPORTER_TYPE", "otlp"),
            sample_rate=float(os.getenv("OTEL_SAMPLE_RATE", "1.0")),
            enabled=os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "development"),
        )


# Global tracer instance
_tracer = None
_tracer_provider = None
_config: Optional[TracingConfig] = None


def init_tracing(config: Optional[TracingConfig] = None) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        config: Tracing configuration. Uses env vars if not provided.
    """
    global _tracer, _tracer_provider, _config

    _config = config or TracingConfig.from_env()

    if not _config.enabled:
        logger.info("Tracing disabled by configuration")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Create resource with service info
        resource = Resource.create(
            {
                "service.name": _config.service_name,
                "service.version": os.getenv("APP_VERSION", "1.0.0"),
                "deployment.environment": _config.environment,
                **_config.extra_attributes,
            }
        )

        # Create sampler
        sampler = TraceIdRatioBased(_config.sample_rate)

        # Create and set tracer provider
        _tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler,
        )

        # Add exporter based on config
        _add_exporter(_config)

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Get tracer
        _tracer = trace.get_tracer(
            _config.service_name, schema_url="https://opentelemetry.io/schemas/1.11.0"
        )

        logger.info(
            f"Tracing initialized: service={_config.service_name}, "
            f"exporter={_config.exporter_type}, "
            f"sample_rate={_config.sample_rate}"
        )

    except ImportError as e:
        logger.warning(f"OpenTelemetry not installed, tracing disabled: {e}")
        _config.enabled = False
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        _config.enabled = False


def _add_exporter(config: TracingConfig) -> None:
    """Add the appropriate span exporter."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor  # noqa:F401

    if config.exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=config.jaeger_endpoint, insecure=True)
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"Added OTLP exporter: {config.jaeger_endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not available, falling back to console")
            _add_console_exporter()

    elif config.exporter_type == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            exporter = JaegerExporter(
                collector_endpoint=config.jaeger_http_endpoint,
            )
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"Added Jaeger exporter: {config.jaeger_http_endpoint}")
        except ImportError:
            logger.warning("Jaeger exporter not available, falling back to console")
            _add_console_exporter()

    elif config.exporter_type == "console":
        _add_console_exporter()

    else:
        logger.warning(f"Unknown exporter type: {config.exporter_type}")
        _add_console_exporter()


def _add_console_exporter() -> None:
    """Add console exporter for debugging."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    _tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    logger.info("Added console exporter")


def get_tracer():
    """Get the global tracer instance."""
    global _tracer

    if _tracer is None:
        # Return a no-op tracer if not initialized
        try:
            from opentelemetry import trace

            return trace.get_tracer("ml-observability-platform")
        except ImportError:
            return _NoOpTracer()

    return _tracer


def shutdown_tracing() -> None:
    """Shutdown tracing and flush pending spans."""
    global _tracer_provider

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("Tracing shutdown complete")


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_span(self, name, **kwargs):
        return _NoOpSpan()

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        yield _NoOpSpan()


class _NoOpSpan:
    """No-op span for when OpenTelemetry is not available."""

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Args:
        name: Span name. Defaults to function name.
        attributes: Static attributes to add to span.

    Example:
        @trace_function(name="process_data", attributes={"operation": "transform"})
        def process_data(df):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name) as span:
                # Add function info
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_prediction(
    model_name: str,
    model_type: str = "unknown",
) -> Callable[[F], F]:
    """Decorator specifically for tracing ML predictions.

    Adds ML-specific attributes to the span.

    Args:
        model_name: Name of the model
        model_type: Type of model (classifier, regressor, etc.)

    Example:
        @trace_prediction(model_name="fraud_detector", model_type="classifier")
        def predict(self, X):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span(f"predict.{model_name}") as span:
                # ML-specific attributes
                span.set_attribute("ml.model.name", model_name)
                span.set_attribute("ml.model.type", model_type)
                span.set_attribute("ml.operation", "predict")

                # Try to get batch size from first positional arg
                if args and len(args) > 1:
                    X = args[1]  # args[0] is self for methods
                    if hasattr(X, "__len__"):
                        span.set_attribute("ml.batch_size", len(X))
                    if hasattr(X, "shape"):
                        span.set_attribute("ml.input_shape", str(X.shape))

                try:
                    result = func(*args, **kwargs)

                    # Add result info
                    if hasattr(result, "__len__"):
                        span.set_attribute("ml.output_size", len(result))

                    span.set_attribute("ml.prediction.success", True)
                    return result

                except Exception as e:
                    span.set_attribute("ml.prediction.success", False)
                    span.set_attribute("ml.prediction.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Context manager for creating a traced span.

    Args:
        name: Span name
        attributes: Attributes to add to span

    Example:
        with trace_span("data_processing", {"stage": "preprocessing"}) as span:
            # do work
            span.set_attribute("rows_processed", 1000)
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def add_span_attributes(attributes: Dict[str, Any]) -> None:
    """Add attributes to the current span.

    Args:
        attributes: Key-value pairs to add

    Example:
        add_span_attributes({
            "ml.drift_detected": True,
            "ml.drift_score": 0.15,
        })
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
    except Exception:
        pass  # Silently ignore if tracing not available


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID as a hex string.

    Useful for correlation with logs and other systems.

    Returns:
        Trace ID as hex string, or None if no active span.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span:
            context = span.get_span_context()
            if context.is_valid:
                return format(context.trace_id, "032x")
    except Exception:
        pass

    return None


def get_current_span_id() -> Optional[str]:
    """Get the current span ID as a hex string."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span:
            context = span.get_span_context()
            if context.is_valid:
                return format(context.span_id, "016x")
    except Exception:
        pass

    return None


# ============================================================================
# FastAPI Integration
# ============================================================================


def setup_fastapi_tracing(app) -> None:
    """Configure OpenTelemetry instrumentation for FastAPI.

    Args:
        app: FastAPI application instance
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")

    except ImportError:
        logger.warning("FastAPI instrumentation not available")


def setup_requests_tracing() -> None:
    """Configure OpenTelemetry instrumentation for requests library."""
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument()
        logger.info("Requests instrumentation enabled")

    except ImportError:
        logger.warning("Requests instrumentation not available")


def setup_httpx_tracing() -> None:
    """Configure OpenTelemetry instrumentation for httpx library."""
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumentation enabled")

    except ImportError:
        logger.warning("HTTPX instrumentation not available")


# ============================================================================
# Logging Integration
# ============================================================================


class TracingLogFilter(logging.Filter):
    """Log filter that adds trace context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_current_trace_id() or "no-trace"
        record.span_id = get_current_span_id() or "no-span"
        return True


def setup_logging_with_trace_context(
    format_string: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Configure logging to include trace context.

    Args:
        format_string: Custom format string. Must include %(trace_id)s and %(span_id)s
        level: Logging level
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s"
        )

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    handler.addFilter(TracingLogFilter())

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)
