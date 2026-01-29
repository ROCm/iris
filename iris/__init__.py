# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
- iris_gluon: Gluon-based implementation with @aggregate backend (experimental)

Quick Start (Traditional API):
    >>> import iris
    >>> ctx = iris.iris(heap_size=2**30)
    >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    >>>
    >>> @triton.jit
    >>> def kernel(buffer, heap_bases):
    >>>     iris.load(buffer, 0, 1, heap_bases)

Quick Start (Gluon API - Experimental):
    >>> import iris.experimental.iris_gluon as iris_gl
    >>> from triton.experimental import gluon
    >>> from triton.experimental.gluon import language as gl
    >>>
    >>> ctx = iris_gl.iris(heap_size=2**30)
    >>> context_tensor = ctx.get_device_context()
    >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    >>>
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     ctx.load(buffer, 1)
"""

# __init__.py

from .iris import (
    Iris,
    iris,
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

from .util import (
    do_bench,
)

from . import hip

# Import experimental features (optional, for users who want experimental APIs)
from . import experimental

# Import ops module (fused GEMM+CCL operations)
from . import ops

# Import logging functionality
from .logging import (
    set_logger_level,
    logger,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
)

# Launcher functionality is now user code - see examples and documentation

__all__ = [
    "Iris",
    "iris",
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
    "hip",
    "experimental",  # Experimental features including iris_gluon
    "ops",  # Fused GEMM+CCL operations
    "set_logger_level",
    "logger",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
]
