# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""Triton device-side context and RMA operations."""

from .context import Context  # noqa: F401
from .context import Context as DeviceContext  # noqa: F401  backward compat
from .tracing import Tracing  # noqa: F401
from .tracing import Tracing as DeviceTracing  # noqa: F401  backward compat
from .types import (  # noqa: F401
    Tile,
    TileView,
    TensorView,
    AllReduceConfig,
    make_tensor_view,
    tile_layout,
    tile_ptr,
    offset_ptr,
    chiplet_transform_chunked,
    compute_tile_indices,
    compute_tile_offsets,
)
from .ops import (  # noqa: F401
    load,
    store,
    copy,
    get,
    put,
    atomic_add,
    atomic_sub,
    atomic_cas,
    atomic_xchg,
    atomic_xor,
    atomic_and,
    atomic_or,
    atomic_min,
    atomic_max,
)
