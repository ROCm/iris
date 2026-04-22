"""Compatibility shim — canonical code lives in iris.host.tracing and iris.device.triton.tracing"""
from iris.host.tracing.events import EVENT_NAMES, TraceEvent  # noqa: F401
from iris.host.tracing.core import Tracing  # noqa: F401
from iris.device.triton.tracing import DeviceTracing  # noqa: F401
from iris.host.tracing import kernel_artifacts  # noqa: F401
