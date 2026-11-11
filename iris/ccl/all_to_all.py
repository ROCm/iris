# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info, _ensure_symmetric, _validate_output_symmetric


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).

    Both tensors are accessed remotely (other ranks read input slices and
    write to output slices via RMA), so both must be on the symmetric heap.

    Args:
        output_tensor: Shape (M, N * world_size) — must be on symmetric heap
        input_tensor: Shape (M, N * world_size) — must be on symmetric heap
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If False (default), end with a device-side barrier on the current
            stream: work queued after this call runs only once every rank's part of
            the collective is done. The host does not wait. If True, skip the
            barrier; the caller must synchronize the ranks, e.g. with
            ``ctx.device_barrier(group=group)``, before reading the output or
            reusing the buffers
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)
    if config.engine != "cu":
        raise ValueError(f"all_to_all does not support engine={config.engine!r}; only all_gather runs on 'sdma'")

    # Input is remote-read by other ranks — auto-import if needed
    input_tensor = _ensure_symmetric(ctx, input_tensor, "input_tensor")
    # Output is remote-written by other ranks — must be pre-allocated on heap
    _validate_output_symmetric(ctx, output_tensor, "output_tensor")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if config.use_gluon and config.all_to_all_variant == "tdm":
        # gfx1250 tensor data movement engine; see gluon/all_to_all_tdm.py
        from iris.ccl.gluon.all_to_all_tdm import launch
    elif config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch
    else:
        from iris.ccl.triton.all_to_all import launch

    launch(
        input_tensor,
        output_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    if not async_op:
        ctx.device_barrier(group=group)
