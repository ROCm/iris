# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Triton only (no gluon support).
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 32 * 1024  # <32KB: NCCL avoids Triton launch overhead
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # >=8MB: NCCL tree-based is more bandwidth-efficient


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None):
    """
    Reduce-scatter: each rank reduces its assigned tiles, stores locally.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    # Fast NCCL dispatch — avoid extract_group_info overhead for small/large messages
    M, N = input_tensor.shape[:2]
    msg_bytes = M * N * input_tensor.element_size()

    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        world_size = _dist.get_world_size(group)
        rank_in_group = _dist.get_rank(group)
        chunk_m = M // world_size
        out_chunk = output_tensor[rank_in_group * chunk_m : (rank_in_group + 1) * chunk_m]
        _dist.reduce_scatter_tensor(out_chunk, input_tensor, group=group)
        return

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

    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant != "two_shot":
        raise ValueError(f"reduce_scatter only supports variant='two_shot', got '{variant}'.")

    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    from iris.ccl.triton.reduce_scatter import launch

    use_inline = not async_op
    barrier_state = None
    if use_inline:
        barrier_state = ctx._get_inline_barrier_state(group)

    launch(
        output_tensor,
        input_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        inline_barrier=use_inline,
        barrier_state=barrier_state,
    )
