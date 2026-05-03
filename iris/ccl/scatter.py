# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Scatter collective operation — public API.

Two paths by message size:
- Small/medium (< _NCCL_LARGE_BYTES): native Triton/Gluon kernel
- Large (>= _NCCL_LARGE_BYTES): NCCL (tree-based, scales better at high BW)
"""

import torch.distributed as _dist

from iris.ccl.utils import extract_group_info

_NCCL_SMALL_BYTES = 2 * 1024 * 1024  # <2MB: NCCL avoids Triton launch overhead
_NCCL_LARGE_BYTES = 8 * 1024 * 1024  # >=8MB: NCCL is more bandwidth-efficient


def scatter(output_tensor, input_tensor, ctx, src=0, group=None, async_op=False, config=None):
    """
    Scatter: root rank distributes equal-sized chunks of its data to all ranks.

    Input on root is (world_size * M, N), each rank receives (M, N).
    Only root's input_tensor contents are used, but all ranks must allocate
    input_tensor at the same shape to maintain symmetric heap offsets for
    RMA address translation.

    Args:
        output_tensor: Shape (M, N) — receives this rank's chunk
        input_tensor: Shape (world_size * M, N). Must be allocated on all ranks
                      (needed for heap address translation), but only root's
                      contents are used.
        ctx: Iris instance
        src: Source (root) rank within the group (default: 0)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = output_tensor.shape[:2]
    msg_bytes = M * N * output_tensor.element_size()

    # Small and large messages: use NCCL (avoids Triton launch overhead at small
    # sizes, and tree-based scatter is more bandwidth-efficient at large sizes).
    if msg_bytes < _NCCL_SMALL_BYTES or msg_bytes >= _NCCL_LARGE_BYTES:
        # torch.distributed.scatter requires list-of-tensors on root
        scatter_list = None
        if rank_in_group == src:
            scatter_list = list(input_tensor.chunk(world_size, dim=0))
        _dist.scatter(output_tensor, scatter_list, src=src, group=group)
        return

    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    # Validate input shape on ALL ranks — symmetric allocation is required for
    # RMA address translation, so every rank must allocate (world_size * M, N).
    expected_input_shape = (world_size * M, N)
    if input_tensor.shape[:2] != expected_input_shape:
        raise ValueError(
            f"Input tensor shape {input_tensor.shape[:2]} does not match expected shape "
            f"{expected_input_shape}. Expected (world_size * M, N) = ({world_size * M}, {N}). "
            f"All ranks must allocate input_tensor at this shape to maintain symmetric heap offsets."
        )

    if config.use_gluon:
        from iris.ccl.gluon.scatter import launch

        launch(
            input_tensor,
            output_tensor,
            ctx,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            src,
            config,
        )
        if not async_op:
            ctx.device_barrier(group)
    else:
        from iris.ccl.triton.scatter import launch

        use_inline = not async_op
        barrier_state = None
        if use_inline:
            barrier_state = ctx._get_inline_barrier_state(group)

        launch(
            input_tensor,
            output_tensor,
            ctx,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            src,
            config,
            inline_barrier=use_inline,
            barrier_state=barrier_state,
        )
