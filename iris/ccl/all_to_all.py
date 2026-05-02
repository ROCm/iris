# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

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
    send_counts,
    send_displs,
    recv_counts,
    recv_displs,
    ctx,
    group=None,
    async_op=False,
    config=None,
    remote_recv_displs=None,
):
    """
    Variable-size all-to-all collective operation.

    Each rank sends send_counts[i] elements starting at send_displs[i] to rank i,
    and receives recv_counts[i] elements from rank i at recv_displs[i].

    Args:
        output_tensor: Output tensor (1D, flat) on symmetric heap.
        input_tensor: Input tensor (1D, flat) on symmetric heap.
        send_counts: list[int] of length world_size -- elements to send to each rank.
        send_displs: list[int] of length world_size -- element offsets in input for each rank.
        recv_counts: list[int] of length world_size -- elements to receive from each rank.
        recv_displs: list[int] of length world_size -- element offsets in output for each rank.
        ctx: Iris context.
        group: ProcessGroup or None.
        async_op: If False, barrier at end.
        config: Config instance.
        remote_recv_displs: Optional list[int] of length world_size. If provided,
            remote_recv_displs[i] is the offset in rank i's output buffer where rank i
            expects to receive data from this rank. Providing this skips an internal
            all_gather call, which can save ~20ms of NCCL overhead. Callers like MoE
            routers that already know the global displacement layout should pass this.

    Note:
        Caller must ensure send_counts[i] on rank A == recv_counts[A] on rank i.
        iris does not validate this.
        Input and output tensors MUST be on the symmetric heap with identical
        allocation sizes across all ranks (symmetric heap invariant).
    """
    import torch

    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    device = input_tensor.device

    # Convert lists to device tensors for kernel access
    send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
    send_displs_t = torch.tensor(send_displs, dtype=torch.int64, device=device)

    if remote_recv_displs is not None:
        # Caller provided pre-computed remote displacements -- skip all_gather.
        kernel_recv_displs_t = torch.tensor(remote_recv_displs, dtype=torch.int64, device=device)
    else:
        # Exchange recv_displs across ranks so each rank knows where to write
        # on every remote rank's output buffer. This uses NCCL all_gather.
        import torch.distributed as dist

        local_recv_displs_t = torch.tensor(recv_displs, dtype=torch.int64, device=device)
        all_recv_displs_list = [torch.zeros(world_size, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(all_recv_displs_list, local_recv_displs_t, group=group)

        # kernel_recv_displs[i] = rank i's recv_displs[group_rank]
        #                       = where rank i stores data from us.
        kernel_recv_displs_t = torch.zeros(world_size, dtype=torch.int64, device=device)
        for i in range(world_size):
            kernel_recv_displs_t[i] = all_recv_displs_list[i][rank_in_group].item()

    # Only Triton path for now (no gluon all_to_all_v)
    from iris.ccl.triton.all_to_all import launch_v

    launch_v(
        input_tensor,
        output_tensor,
        send_counts_t,
        send_displs_t,
        kernel_recv_displs_t,
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
