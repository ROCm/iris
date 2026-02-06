"""
Device-side tracing support for Iris.

Provides tracing functionality to capture and export device-side events
for debugging and performance analysis.
"""

from .events import EVENT_NAMES, TraceEvent
from .core import Tracing
from .device import DeviceTracing

__all__ = ["EVENT_NAMES", "TraceEvent", "Tracing", "DeviceTracing"]
