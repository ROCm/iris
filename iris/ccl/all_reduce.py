# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Drop-in replacement for torch.distributed.all_reduce.
Accepts regular CUDA tensors, handles heap copy internally.
Graph-capture safe with in-kernel barriers (one_shot variant).
"""

from iris.ccl.utils import extract_group_info

_buf_cache = {}


def _get_heap_bufs(ctx, shape, dtype):
    """Get or create cached heap buffers for stable graph capture addresses."""
    key = (shape, dtype)
    if key not in _buf_cache:
        _buf_cache[key] = {
            "inp": ctx.zeros(shape, dtype=dtype),
            "out": ctx.zeros(shape, dtype=dtype),
        }
    return _buf_cache[key]["inp"], _buf_cache[key]["out"]


def _is_on_heap(tensor, ctx):
    """Check if tensor is allocated on the iris symmetric heap."""
    try:
        heap = ctx.heap
        ptr = tensor.data_ptr()
        base = heap.base_addr
        size = heap.heap_size
        return base <= ptr < base + size
    except Exception:
        return False


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    if not _is_on_heap(input_tensor, ctx):
        heap_inp, heap_out = _get_heap_bufs(ctx, input_tensor.shape, input_tensor.dtype)
        return _preamble(heap_out, heap_inp, ctx, config=config, workspace=workspace)
    return _preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

    Accepts regular CUDA tensors — copies to/from symmetric heap internally.
    Graph-capture safe when using one_shot variant (in-kernel barriers).

    Args:
        output_tensor: Shape (M, N) — receives the reduced result
        input_tensor: Shape (M, N) — local rank's partial data
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace from all_reduce_preamble
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
        config = Config(all_reduce_variant="one_shot")

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot", "one_shot_legacy"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    needs_copy = not _is_on_heap(input_tensor, ctx)
    if needs_copy:
        heap_inp, heap_out = _get_heap_bufs(ctx, input_tensor.shape, input_tensor.dtype)
        heap_inp.copy_(input_tensor)
        kernel_in, kernel_out = heap_inp, heap_out
    else:
        kernel_in, kernel_out = input_tensor, output_tensor

    from iris.ccl.triton.all_reduce import launch

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
        workspace,
        group=group,
    )

    if needs_copy:
        output_tensor.copy_(kernel_out)

    if workspace is not None:
        if variant != "one_shot":
            workspace.prepared = False

    if not async_op and variant != "one_shot":
        ctx.barrier()

    return workspace
