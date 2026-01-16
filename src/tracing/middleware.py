"""FastAPI middleware for distributed tracing."""

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .tracer import add_span_attributes, get_current_trace_id, get_tracer, trace_span  # noqa:F401

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds tracing to all HTTP requests.

    Creates a span for each request with:
    - HTTP method and path
    - Status code
    - Duration
    - Client IP
    - User agent
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with tracing."""
        tracer = get_tracer()

        # Extract useful request info
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Create span name
        span_name = f"{method} {path}"

        with tracer.start_as_current_span(span_name) as span:
            # Add request attributes
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", path)
            span.set_attribute("http.client_ip", client_host)
            span.set_attribute("http.user_agent", user_agent)
            span.set_attribute("http.scheme", request.url.scheme)

            # Add trace ID to request state for logging
            request.state.trace_id = get_current_trace_id()

            start_time = time.perf_counter()

            try:
                response = await call_next(request)

                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute(
                    "http.response_content_type", response.headers.get("content-type", "unknown")
                )

                # Add trace ID to response headers
                trace_id = get_current_trace_id()
                if trace_id:
                    response.headers["X-Trace-ID"] = trace_id

                # Mark success/failure based on status code
                if response.status_code >= 400:
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", "http_error")

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                span.record_exception(e)
                raise

            finally:
                duration = time.perf_counter() - start_time
                span.set_attribute("http.duration_ms", duration * 1000)


class MLPredictionTracingMiddleware(BaseHTTPMiddleware):
    """Specialized middleware for ML prediction endpoints.

    Adds ML-specific attributes to prediction requests.
    """

    PREDICTION_PATHS = {"/predict/fraud", "/predict/price", "/predict/churn", "/predict/batch"}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process ML prediction requests with enhanced tracing."""
        path = request.url.path

        if path not in self.PREDICTION_PATHS:
            return await call_next(request)

        tracer = get_tracer()

        # Determine model name from path
        model_name = path.split("/")[-1]
        if model_name == "batch":
            model_name = "batch_prediction"

        span_name = f"ml.predict.{model_name}"

        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("ml.model.name", model_name)
            span.set_attribute("ml.operation", "predict")

            # Try to get batch size from request body
            try:
                body = await request.json()
                if isinstance(body, list):
                    span.set_attribute("ml.batch_size", len(body))
                elif "samples" in body:
                    span.set_attribute("ml.batch_size", len(body["samples"]))
                else:
                    span.set_attribute("ml.batch_size", 1)
            except Exception:
                span.set_attribute("ml.batch_size", 1)

            start_time = time.perf_counter()

            try:
                response = await call_next(request)

                duration = time.perf_counter() - start_time
                span.set_attribute("ml.prediction.duration_ms", duration * 1000)
                span.set_attribute("ml.prediction.success", response.status_code < 400)

                return response

            except Exception as e:
                span.set_attribute("ml.prediction.success", False)
                span.set_attribute("ml.prediction.error", str(e))
                span.record_exception(e)
                raise


def add_tracing_to_app(app) -> None:
    """Add all tracing middleware to FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Add middlewares (order matters - first added = outermost)
    app.add_middleware(MLPredictionTracingMiddleware)
    app.add_middleware(TracingMiddleware)

    logger.info("Tracing middleware added to FastAPI app")
