# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.compile: torch.compile-compatible functional collectives.

Registers iris collective operations as custom ops via torch.library so that
torch.compile (inductor backend) can trace through them without graph breaks.

Usage::

    import iris.compile  # registers ops on import
    import torch

    ctx = iris.iris(heap_size=2**30)
    iris.compile.set_context(ctx)

    @torch.compile
    def my_fn(x):
        return torch.ops.iris.all_reduce(x)
"""

from iris.compile.functional import set_context, get_context  # noqa: F401
