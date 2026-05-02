# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Supports two variants:
- two_shot: Each rank reads all other ranks' data for its assigned tiles (all-pairs read).
- ring_chunked: Ring-based Rabenseifner-style reduce-scatter with flag-based synchronization.
"""

from iris.ccl.utils import extract_group_info


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
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
        workspace: ReduceScatterWorkspace for reusing ring buffers across calls.
                   Only used by ring_chunked variant. If None, allocated internally.
    """
    from iris.ccl.utils import ReduceOp

    if op is None:
        op = ReduceOp.SUM
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )
    if config is None:
        # Adaptive defaults tuned on MI308X x 4 vs RCCL.
        # Key insights:
        # - bm=16 creates more tiles for better SM utilization at all sizes.
        # - comm_sms sweet spot depends on tensor size and XGMI contention:
        #   - 128M+: sms=64 needed for large tile counts
        #   - 64M: sms=48 reduces XGMI link contention (94% roofline at peak fclk)
        #   - 4M-64M: sms=32 optimal — enough tiles for parallelism,
        #     fewer SMs reduces XGMI contention
        #   - <4M: sms=48 — too few tiles for sms=32 to saturate
        # Uses _default_config() cache to avoid repeated HIP subprocess queries.
        from iris.ccl.triton.reduce_scatter import _default_config

        M_in, N_in = input_tensor.shape[:2]
        total_elems = M_in * N_in
        if total_elems >= 128 * 1024 * 1024:  # >= 128M elements
            config = _default_config(16, 64)  # comm_sms=64 default
        elif total_elems >= 64 * 1024 * 1024:  # >= 64M elements
            config = _default_config(16, 64, comm_sms=48)
        elif total_elems >= 4 * 1024 * 1024:  # >= 4M elements
            config = _default_config(16, 64, comm_sms=32)
        else:
            config = _default_config(16, 64, comm_sms=48)

    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    variant = getattr(config, "reduce_scatter_variant", "two_shot")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    M, N = input_tensor.shape[:2]

    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    from iris.ccl.triton.reduce_scatter import launch

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
        workspace=workspace,
    )

    if not async_op:
        ctx.barrier()
