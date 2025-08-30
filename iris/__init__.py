# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# __init__.py

import os
import torch

# Always import logging functionality (no MPI dependency)
from .logging import (
    set_logger_level,
    logger,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
)

# Try to import main functionality - may fail if MPI is not available
_IRIS_CORE_AVAILABLE = False
try:
    from .iris import (
        Iris,
        iris,
        load,
        store,
        get,
        put,
        atomic_add,
        atomic_sub,
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
        memset_tensor,
    )

    from . import hip

    _IRIS_CORE_AVAILABLE = True

except Exception as e:
    # If MPI or other dependencies are not available, only provide logging functionality
    import warnings

    warnings.warn(
        f"Iris core functionality not available due to missing dependencies: {e}. "
        "Only logging functionality is available.",
        ImportWarning,
    )

# Set up the __all__ list based on what's available
if _IRIS_CORE_AVAILABLE:
    __all__ = [
        "Iris",
        "iris",
        "load",
        "store",
        "get",
        "put",
        "atomic_add",
        "atomic_sub",
        "atomic_cas",
        "atomic_xchg",
        "atomic_xor",
        "atomic_or",
        "atomic_and",
        "atomic_min",
        "atomic_max",
        "do_bench",
        "memset_tensor",
        "hip",
        "set_logger_level",
        "logger",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ]

    # Pipe allocations via finegrained allocator
    current_dir = os.path.dirname(__file__)
    # Look for the library in the installed package location
    finegrained_alloc_path = os.path.join(current_dir, "..", "csrc", "finegrained_alloc", "libfinegrained_allocator.so")

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
else:
    # Set up minimal __all__ list with only logging functionality
    __all__ = [
        "set_logger_level",
        "logger",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ]
