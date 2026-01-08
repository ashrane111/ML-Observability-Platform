"""OpenTelemetry tracing module for ML Observability Platform."""

from .tracer import (
    TracingConfig,
    init_tracing,
    get_tracer,
    trace_function,
    trace_prediction,
    add_span_attributes,
    get_current_trace_id,
    shutdown_tracing,
)

__all__ = [
    "TracingConfig",
    "init_tracing",
    "get_tracer",
    "trace_function",
    "trace_prediction",
    "add_span_attributes",
    "get_current_trace_id",
    "shutdown_tracing",
]