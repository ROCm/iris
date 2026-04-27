# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective communication primitive for Iris.
Supports both Triton and Gluon implementations based on config.
"""

from .config import Config
from .utils import extract_group_info


def all_to_all(
    output_tensor,
    input_tensor,
    shmem,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal all-to-all collective operation implementation.

    This function is called internally by shmem.ccl.all_to_all().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.all_to_all(output_tensor, input_tensor)

    Each rank sends a tensor chunk to each other rank and receives
    a tensor chunk from each other rank. Input/output tensors should have
    shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

    Args:
        output_tensor: Output tensor of shape (M, N * world_size)
        input_tensor: Input tensor of shape (M, N * world_size)
        shmem: Iris context
        group: ProcessGroup or None. If None, uses all ranks in shmem context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.use_gluon=True to use Gluon implementation with traffic shaping.
    """
    # Use provided config or create default one
    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    # Extract group information
    # rank_in_group: position within the ProcessGroup (0, 1, 2, ...) - passed as group_rank to kernel
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    if config.use_gluon:
        from .gluon import GLUON_AVAILABLE

        if not GLUON_AVAILABLE:
            raise ValueError("Gluon is not available. Install Triton with Gluon support or set use_gluon=False")
        from .gluon.all_to_all import dispatch_gluon

        dispatch_gluon(
            input_tensor,
            output_tensor,
            shmem,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config,
        )
    else:
        from .triton.all_to_all import dispatch_triton

        dispatch_triton(
            input_tensor,
            output_tensor,
            shmem,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config,
        )

    if not async_op:
        shmem.barrier()
