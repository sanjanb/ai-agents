"""Simple OpenTelemetry tracing setup for FastAPI or scripts.

Usage:
    from tracing_utils import setup_tracer
    tracer = setup_tracer(service_name="agent-api")
"""

from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


_configured = False


def setup_tracer(service_name: str = "agent-service", endpoint: Optional[str] = None):
    global _configured
    if _configured:
        return trace.get_tracer(service_name)

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _configured = True
    return trace.get_tracer(service_name)


def traced(tracer=None, name: str = "operation"):
    tracer = tracer or trace.get_tracer("default")

    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
