# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective communication primitive for Iris.
Supports multiple variants: atomic, spinlock, ring, two-shot, and one-shot.
"""

from typing import Optional

from .config import Config
from .utils import ReduceOp, extract_group_info

# Re-export for external consumers
from .triton.all_reduce import AllReduceWorkspace, all_reduce_preamble  # noqa: F401


def all_reduce(
    output_tensor,
    input_tensor,
    shmem,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config=None,
    workspace: Optional[AllReduceWorkspace] = None,
):
    """
    Internal all-reduce collective operation implementation.

    This function is called internally by shmem.ccl.all_reduce().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.all_reduce(output_tensor, input_tensor)

    Each rank has a local input tensor, and all ranks compute the sum of all
    input tensors. The result is written to output_tensor on all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain sum of all inputs
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
                Set config.all_reduce_variant to choose variant: "atomic", "spinlock", "ring", "two_shot", or "one_shot"
        workspace: Optional AllReduceWorkspace instance prepared via all_reduce_preamble.
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
            "all_reduce does not support use_gluon=True. "
            "Gluon implementation is not available for all_reduce. "
            "Use default config (use_gluon=False)."
        )

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    # Extract group information
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    from .triton.all_reduce import dispatch_triton

    workspace = dispatch_triton(
        output_tensor,
        input_tensor,
        shmem,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        workspace,
        group=group,
    )

    if workspace is not None:
        workspace.prepared = False

    if not async_op:
        shmem.barrier()

    return workspace
