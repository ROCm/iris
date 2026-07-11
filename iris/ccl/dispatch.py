# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Shared dispatch helpers for iris CCL collectives.

Provides heap buffer caching for stable graph capture addresses
and automatic copy between user tensors and symmetric heap.
"""

_buf_cache = {}


def _cache_key(shape, dtype, prefix):
    return (prefix, shape, dtype)


def get_heap_buffer(ctx, shape, dtype, prefix):
    """Get or create a cached heap buffer with stable address for graph capture."""
    key = _cache_key(shape, dtype, prefix)
    if key not in _buf_cache:
        _buf_cache[key] = ctx.zeros(shape, dtype=dtype)
    buf = _buf_cache[key]
    if buf.shape != shape or buf.dtype != dtype:
        _buf_cache[key] = ctx.zeros(shape, dtype=dtype)
        buf = _buf_cache[key]
    return buf


def is_on_heap(tensor, ctx):
    """Check if a tensor is already on the symmetric heap."""
    heap_base = ctx.heap.get_base_address()
    ptr = tensor.data_ptr()
    return heap_base <= ptr < heap_base + ctx.heap_size
