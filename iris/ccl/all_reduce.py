# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Triton only (no gluon support).
"""

from iris.ccl.utils import extract_group_info, _ensure_symmetric, _validate_output_symmetric


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

    Which tensors need the symmetric heap depends on the variant:
    - atomic/spinlock: output is remote-accessed (input is local)
    - one_shot: input is remote-read (output is local)
    - two_shot: both are remote-accessed
    - ring: neither (workspace ring_buffer/flags are, but those are internal)

    We ensure symmetric on the tensors that each variant actually accesses
    remotely. Workspace tensors (ring_buffer, flags, locks) are allocated
    internally on the heap already.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
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

    # Ensure/validate symmetric only for tensors that are remotely accessed per variant
    if variant in ("one_shot", "two_shot"):
        # Input is remote-read — auto-import if needed
        input_tensor = _ensure_symmetric(ctx, input_tensor, "input_tensor")
    if variant in ("atomic", "spinlock", "two_shot"):
        # Output is remote-written — must be pre-allocated on heap
        _validate_output_symmetric(ctx, output_tensor, "output_tensor")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    from iris.ccl.triton.all_reduce import launch

    workspace = launch(
        output_tensor,
        input_tensor,
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

    if workspace is not None:
        workspace.prepared = False

    if not async_op:
        ctx.barrier()

    return workspace
