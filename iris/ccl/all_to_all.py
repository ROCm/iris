# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operations — public API.

Provides:
  - all_to_all: Fixed-size per-rank chunks (matches torch.distributed.all_to_all)
  - all_to_all_v: Variable-size per-rank chunks (matches torch.distributed.all_to_all_single with split_sizes)

Routes to triton/ or gluon/ based on config.use_gluon.
"""

import torch
from iris.ccl.utils import extract_group_info


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).

    Args:
        output_tensor: Shape (M, N * world_size)
        input_tensor: Shape (M, N * world_size)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if config.use_gluon:
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
        ctx.barrier()


def all_to_all_v(
    output_tensor,
    input_tensor,
    output_split_sizes,
    input_split_sizes,
    ctx,
    group=None,
    async_op=False,
    config=None,
):
    """
    All-to-all with variable-size per-rank chunks.

    Each rank sends input_split_sizes[i] columns to rank i, and receives
    output_split_sizes[i] columns from rank i. This matches the semantics of
    torch.distributed.all_to_all_single with split_sizes.

    Note:
        This function requires an initialized ``torch.distributed`` process group
        for the offset metadata exchange (a small all-to-all of int64 offsets).
        The bulk data transfer uses iris's symmetric heap.

    Args:
        output_tensor: Shape (M, sum(output_split_sizes))
        input_tensor: Shape (M, sum(input_split_sizes))
        output_split_sizes: List[int] of length world_size — columns to receive from each rank
        input_split_sizes: List[int] of length world_size — columns to send to each rank
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    import torch.distributed as dist

    device = input_tensor.device

    # Convert split sizes to tensors on device
    input_split_sizes_t = torch.tensor(input_split_sizes, dtype=torch.int64, device=device)

    # Compute local input offsets (cumulative sum with 0 prepended)
    input_split_offsets_t = torch.zeros(world_size, dtype=torch.int64, device=device)
    if world_size > 1:
        input_split_offsets_t[1:] = torch.cumsum(input_split_sizes_t[:-1], dim=0)

    # Compute remote output offsets: for each rank i, where in rank i's output
    # buffer should our data be placed? This requires knowing rank i's
    # output_split_sizes and computing the offset for slot group_rank.
    # We use an all-to-all exchange of offsets to compute this.
    output_split_sizes_t = torch.tensor(output_split_sizes, dtype=torch.int64, device=device)
    output_split_offsets_t = torch.zeros(world_size, dtype=torch.int64, device=device)
    if world_size > 1:
        output_split_offsets_t[1:] = torch.cumsum(output_split_sizes_t[:-1], dim=0)

    # remote_output_offsets[i] = rank i's output offset for data from group_rank
    # Each rank broadcasts its output_split_offsets, and we pick the slot for group_rank.
    # Use all_to_all on the offsets themselves:
    # Each rank sends its output_split_offsets[j] to rank j (the offset where rank j should write).
    remote_output_offsets_t = torch.zeros(world_size, dtype=torch.int64, device=device)
    # Exchange: each rank sends output_split_offsets to all ranks
    # output_split_offsets[j] is where rank j's data lands in OUR output buffer
    # So rank j needs output_split_offsets[j] from us.
    # We send our output_split_offsets, each rank picks slot group_rank from what it receives.
    send_list = [output_split_offsets_t[i : i + 1].clone() for i in range(world_size)]
    recv_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_to_all(recv_list, send_list, group=group)
    for i in range(world_size):
        remote_output_offsets_t[i] = recv_list[i][0]

    # AllToAllv is currently Triton-only; raise if gluon requested
    if config.use_gluon:
        raise NotImplementedError("AllToAllv is not yet implemented for the gluon backend")
    from iris.ccl.triton.all_to_all import launch_v

    launch_v(
        input_tensor,
        output_tensor,
        input_split_sizes_t,
        input_split_offsets_t,
        remote_output_offsets_t,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    if not async_op:
        ctx.barrier()
