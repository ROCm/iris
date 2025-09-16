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
- Memory operations: load, store, get, put
- Utility functions: do_bench
- HIP integration for AMD GPU support
- Logging utilities with rank information

Quick Start:
    >>> import iris
    >>> ctx = iris.iris(heap_size=2**30)
    >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
"""

# __init__.py

import os
import torch

from .iris import (
    Iris,
    iris,
    load,
    store,
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

# Pipe allocations via finegrained allocator
current_dir = os.path.dirname(__file__)
# Look for the library in the installed package location
finegrained_alloc_path = os.path.join(current_dir, "csrc", "finegrained_alloc", "libfinegrained_allocator.so")

# Check if the library exists (should be built during pip install)
if not os.path.exists(finegrained_alloc_path):
    raise RuntimeError(
        f"Fine-grained allocator library not found at {finegrained_alloc_path}. "
        "Please ensure the package was installed correctly."
    )

finegrained_allocator = torch.cuda.memory.CUDAPluggableAllocator(
    finegrained_alloc_path,
    "finegrained_hipMalloc",
    "finegrained_hipFree",
)
torch.cuda.memory.change_current_allocator(finegrained_allocator)

__all__ = [
    "Iris",
    "iris",
    "load",
    "store",
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
    "set_logger_level",
    "logger",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
]
