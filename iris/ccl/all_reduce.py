# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Triton only (no gluon support).
"""

from iris.ccl.utils import extract_group_info

# Cached default configs for auto-variant selection.
# Config() is expensive (~14ms due to hip.get_num_xcc()), so we construct once.
_default_configs = {}


def _get_default_config(variant):
    """Return a cached default Config for the given variant."""
    if variant not in _default_configs:
        from iris.ccl.config import Config

        _default_configs[variant] = Config(
            all_reduce_variant=variant,
            block_size_m=32,
            block_size_n=64,
            all_reduce_distribution=1,
        )
    return _default_configs[variant]


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

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
    from iris.ccl.utils import ReduceOp

    if op is None:
        op = ReduceOp.SUM
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )
    if config is None:
        # Auto-select variant based on message size.
        # Benchmark data (MI300X, 8 GPUs, bf16, N=2880):
        #   all_pairs_chunked: wins at M≤128 (every rank reads all others, barrier-free)
        #   two_shot: wins at M≥256 (Rabenseifner reduce-scatter + all-gather)
        # Crossover at ~720KB total message size.
        msg_bytes = output_tensor.nelement() * output_tensor.element_size()
        variant = "all_pairs_chunked" if msg_bytes <= 768 * 1024 else "two_shot"
        config = _get_default_config(variant)
    if config.use_gluon:
        raise ValueError(
            "all_reduce does not support use_gluon=True. "
            "Gluon implementation is not available for all_reduce. "
            "Use default config (use_gluon=False)."
        )

    variant = config.all_reduce_variant.lower()
    if variant == "auto":
        msg_bytes = output_tensor.nelement() * output_tensor.element_size()
        if msg_bytes <= 768 * 1024:
            variant = "all_pairs_chunked"
        else:
            variant = "two_shot"
        config.all_reduce_variant = variant
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot", "all_pairs_chunked", "auto"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

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
        # all_pairs_chunked only writes to local output (no remote stores),
        # so a post-kernel barrier is unnecessary — cuda stream ordering
        # already guarantees local writes are visible to subsequent ops.
        if variant != "all_pairs_chunked":
            ctx.device_barrier(group=group)

    return workspace
