# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Triton only (no gluon support). Accepts user tensors directly —
copies to symmetric heap internally if the input is not already
on the heap.
"""

from iris.ccl.utils import extract_group_info

_buf_cache = {}
_ws_cache = {}


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)


def _get_heap_bufs(ctx, shape, dtype):
    """Get or create cached heap buffers for a (shape, dtype) pair."""
    key = (shape, dtype)
    if key not in _buf_cache:
        _buf_cache[key] = (ctx.empty(shape, dtype=dtype), ctx.empty(shape, dtype=dtype))
    return _buf_cache[key]


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

    If input_tensor is not on the symmetric heap, it is automatically
    copied to a heap buffer before the collective. The result is returned
    in a heap-resident output buffer (stable address for graph capture).

    Args:
        output_tensor: Shape (M, N), or None to auto-allocate on heap
        input_tensor: Shape (M, N) — can be any CUDA tensor (heap or user)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace from all_reduce_preamble

    Returns:
        workspace object (output data is in output_tensor or heap buffer)
    """
    from iris.ccl.config import Config
    from iris.ccl.utils import ReduceOp

    if op is None:
        op = ReduceOp.SUM
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )
    if config is None:
        config = Config(
            all_reduce_variant="two_shot",
            block_size_m=32,
            block_size_n=64,
            all_reduce_distribution=1,
        )
    if config.use_gluon:
        raise ValueError(
            "all_reduce does not support use_gluon=True. "
            "Gluon implementation is not available for all_reduce. "
            "Use default config (use_gluon=False)."
        )

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    on_heap = ctx.heap.on_symmetric_heap(input_tensor)

    if on_heap:
        inp = input_tensor
        out = output_tensor
    else:
        inp_buf, out_buf = _get_heap_bufs(ctx, input_tensor.shape, input_tensor.dtype)
        inp_buf.copy_(input_tensor)
        inp = inp_buf
        out = out_buf

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    from iris.ccl.triton.all_reduce import launch

    ws_key = (input_tensor.shape, input_tensor.dtype)
    ws = _ws_cache.get(ws_key, workspace)

    ws = launch(
        out,
        inp,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        ws,
        group=group,
    )

    _ws_cache[ws_key] = ws

    if ws is not None:
        ws.prepared = False

    if not async_op:
        ctx.barrier()

    if not on_heap and output_tensor is not None:
        output_tensor.copy_(out)

    return ws, out
