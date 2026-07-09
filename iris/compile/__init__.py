# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.compile: torch.compile integration for iris collective operations.

This module registers iris collective operations as custom operators compatible
with torch.compile tracing, AOTAutograd, and fake tensor mode. It enables
iris collectives to be used within torch.compile'd functions and models.

Usage:
    >>> import iris
    >>> from iris.compile import functional_collectives as fc
    >>>
    >>> ctx = iris.iris(heap_size=2**30)
    >>>
    >>> # Use functional collectives with torch.compile
    >>> @torch.compile
    ... def my_model(x):
    ...     return fc.all_reduce(x, ctx)
    >>>
    >>> output = my_model(input_tensor)
"""

from iris.compile.functional_collectives import (
    IrisCompileContext,
    all_reduce,
    all_gather,
    reduce_scatter,
    setup,
)

__all__ = [
    "IrisCompileContext",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "setup",
]
