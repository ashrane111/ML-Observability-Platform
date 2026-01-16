"""OpenTelemetry tracing module for ML Observability Platform."""

from .tracer import (
    TracingConfig,
    add_span_attributes,
    get_current_trace_id,
    get_tracer,
    init_tracing,
    shutdown_tracing,
    trace_function,
    trace_prediction,
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
