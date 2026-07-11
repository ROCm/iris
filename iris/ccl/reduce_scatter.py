# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Drop-in replacement for torch.distributed.reduce_scatter_tensor.
Accepts regular CUDA tensors, handles heap copy internally.
"""

from iris.ccl.utils import extract_group_info

_buf_cache = {}


def _get_heap_bufs(ctx, shape, dtype):
    key = ("rs", shape, dtype)
    if key not in _buf_cache:
        _buf_cache[key] = {
            "inp": ctx.zeros(shape, dtype=dtype),
            "out": ctx.zeros(shape, dtype=dtype),
        }
    return _buf_cache[key]["inp"], _buf_cache[key]["out"]


def _is_on_heap(tensor, ctx):
    try:
        heap = ctx.heap
        ptr = tensor.data_ptr()
        base = heap.base_addr
        size = heap.heap_size
        return base <= ptr < base + size
    except Exception:
        return False


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    Reduce-scatter: each rank reduces its assigned tiles, stores locally.

    Accepts regular CUDA tensors — copies to/from symmetric heap internally.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace (for twophase variant scratch buffer)
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
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    needs_copy = not _is_on_heap(input_tensor, ctx)
    if needs_copy:
        heap_inp, heap_out = _get_heap_bufs(ctx, input_tensor.shape, input_tensor.dtype)
        heap_inp.copy_(input_tensor)
        kernel_in, kernel_out = heap_inp, heap_out
    else:
        kernel_in, kernel_out = input_tensor, output_tensor

    from iris.ccl.triton.reduce_scatter import launch

    workspace = launch(
        kernel_out,
        kernel_in,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        workspace=workspace,
    )

    if needs_copy:
        output_tensor.copy_(kernel_out)

    if not async_op:
        ctx.barrier()

    return workspace
