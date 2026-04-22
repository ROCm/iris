# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""Compatibility shim — canonical code lives in iris.host.iris, iris.device.triton.*"""

# Host class
from iris.host.iris import Iris, iris

# Device context
from iris.device.triton.context import DeviceContext, __translate

# Device ops
from iris.device.triton.ops import (
    load, store, copy, get, put,
    atomic_add, atomic_sub, atomic_cas, atomic_xchg,
    atomic_xor, atomic_and, atomic_or, atomic_min, atomic_max,
)

# Re-export TraceEvent (originally imported through iris.iris)
from iris.host.tracing.events import TraceEvent

__all__ = [
    "Iris", "iris", "DeviceContext", "TraceEvent", "__translate",
    "load", "store", "copy", "get", "put",
    "atomic_add", "atomic_sub", "atomic_cas", "atomic_xchg",
    "atomic_xor", "atomic_and", "atomic_or", "atomic_min", "atomic_max",
]
