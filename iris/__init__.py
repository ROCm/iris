# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# __init__.py

import os
import torch

from .iris import (
    Iris,
    iris,
    translate,
    load,
    store,
    get,
    put,
    atomic_add,
    atomic_sub,
    atomic_cas,
    atomic_xchg,
)

from .util import (
    do_bench,
    memset_tensor,
)

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
    "translate",
    "load",
    "store",
    "get",
    "put",
    "atomic_add",
    "atomic_sub",
    "atomic_cas",
    "atomic_xchg",
    "do_bench",
    "memset_tensor",
]
