# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris: Multi-GPU Communication and Memory Management Framework

Iris is a high-performance framework for multi-GPU communication and memory management,
providing efficient distributed tensor operations, atomic operations, and memory allocation
across multiple GPUs in a cluster.

This package provides:
- Iris: Main class for multi-GPU operations
- Atomic operations: add, sub, cas, xchg, xor, and, or, min, max
- Memory operations: load, store, copy, get, put
- Utility functions: do_bench
- HIP integration for AMD GPU support
- Logging utilities with rank information
- Gluon backend: @aggregate-based implementation via iris.mem.gluon

Quick Start (Traditional API):
    >>> import iris
    >>> ctx = iris.iris(heap_size=2**30)
    >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    >>>
    >>> @triton.jit
    >>> def kernel(buffer, heap_bases):
    >>>     iris.load(buffer, 0, 1, heap_bases)

Quick Start (Gluon API - Experimental):
    >>> import iris
    >>> from iris.gluon import IrisDeviceCtx
    >>> from triton.experimental import gluon
    >>> from triton.experimental.gluon import language as gl
    >>>
    >>> ctx = iris.iris(heap_size=2**30)
    >>> context_tensor = ctx.get_device_context()
    >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    >>>
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     ctx.load(buffer, 1)
"""

from iris.host.iris import Iris, iris
from iris.mem.triton.context import Context, Context as DeviceContext
from iris.host.tracing.events import TraceEvent
from iris.mem.triton.types import (
    Tile,
    TileView,
    TensorView,
    AllReduceConfig,
    make_tensor_view,
)
from iris.mem.triton.ops import (
    load,
    store,
    copy,
    get,
    put,
    atomic_add,
    atomic_cas,
    atomic_xchg,
    atomic_xor,
    atomic_or,
    atomic_and,
    atomic_min,
    atomic_max,
)

from iris.host.platform.utils import (
    do_bench,
    get_device_id_for_rank,
    is_simulation_env,
)

from iris.host.memory.tensor_utils import (
    CUDAArrayInterface,
    tensor_from_ptr,
)

from iris.host.platform import hip
from . import experimental
from . import ops
from iris.host.memory import tensors as tensor_creation
from . import bench
from iris.host.tracing import kernel_artifacts  # noqa: F401  # triggers _init() at import time
from iris.host.logging.logging import (
    set_logger_level,
    logger,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
)

__all__ = [
    "Iris",
    "iris",
    "get_device_id_for_rank",
    "Context",
    "DeviceContext",
    "TraceEvent",
    "Tile",
    "TileView",
    "TensorView",
    "AllReduceConfig",
    "make_tensor_view",
    "load",
    "store",
    "copy",
    "get",
    "put",
    "atomic_add",
    "atomic_cas",
    "atomic_xchg",
    "atomic_xor",
    "atomic_or",
    "atomic_and",
    "atomic_min",
    "atomic_max",
    "do_bench",
    "CUDAArrayInterface",
    "tensor_from_ptr",
    "hip",
    "experimental",
    "ops",
    "tensor_creation",
    "bench",
    "set_logger_level",
    "logger",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
]

# Patch torch.cuda.set_device to automatically handle device ID wrapping in simulation mode
# Only patch if in simulation mode
if is_simulation_env():
    import torch

    _original_set_device = torch.cuda.set_device

    def _patched_set_device(device):
        """Patched version of torch.cuda.set_device that wraps device IDs in simulation mode."""
        num_devices = torch.cuda.device_count()
        if num_devices > 0 and isinstance(device, int) and device >= num_devices:
            device = device % num_devices
        return _original_set_device(device)

    torch.cuda.set_device = _patched_set_device
