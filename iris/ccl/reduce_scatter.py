# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective communication primitive for Iris.
Uses the two-shot approach: reduce assigned tiles and store only to own rank.
"""

from .config import Config
from .utils import ReduceOp, extract_group_info


def reduce_scatter(
    output_tensor,
    input_tensor,
    shmem,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal reduce-scatter collective operation implementation.

    This function is called internally by shmem.ccl.reduce_scatter().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

    Each rank reduces its assigned tiles from all ranks' inputs and stores
    the result only to its own output tensor. This is similar to all-reduce
    but without broadcasting the result to all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
        input_tensor: Input tensor of shape (M, N) - local rank's partial data
        shmem: Iris shmem context
        op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
            Default: ReduceOp.SUM.
        group: ProcessGroup or None. If None, uses all ranks in shmem context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Only supports reduce_scatter_variant="two_shot".

    Example:
        >>> shmem = iris.iris()
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

        >>> # Custom configuration
        >>> from iris.ccl import Config
        >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    """
    # Validate op parameter
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations (PRODUCT, MAX, MIN, etc.) will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    # Validate that only two_shot variant is used
    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant != "two_shot":
        raise ValueError(
            f"reduce_scatter only supports variant='two_shot', got '{variant}'. "
            f"Set config.reduce_scatter_variant='two_shot' or use default config."
        )

    # Extract group information
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)
    M, N = input_tensor.shape[:2]

    # Validate output shape matches input shape
    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    from .triton.reduce_scatter import dispatch_triton

    dispatch_triton(
        output_tensor,
        input_tensor,
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
