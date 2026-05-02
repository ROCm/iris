# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Point-to-point send/recv operations — public API.

Uses the same pattern as all_gather: write data via iris.store,
then synchronize with ctx.barrier() on the host side.
"""

from iris.ccl.utils import extract_group_info


def send(tensor, ctx, dst, group=None, tag=0, config=None):
    """
    Send tensor to destination rank.

    Writes data to the destination rank's buffer via XGMI remote stores,
    then does a host-side barrier to ensure completion.

    Args:
        tensor: Tensor to send, shape (M, N). Must be on symmetric heap.
        ctx: Iris instance.
        dst: Destination rank within the group.
        group: ProcessGroup or None.
        tag: Communication tag (default: 0). Currently unused.
        config: Config with kernel parameters (default: None).
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if dst < 0 or dst >= world_size:
        raise ValueError(f"dst rank {dst} is out of range for world_size {world_size}")
    if dst == rank_in_group:
        raise ValueError("Cannot send to self — use tensor.copy_() instead")

    dst_iris_rank = rank_start + dst * rank_stride

    from iris.ccl.triton.sendrecv import launch_send

    launch_send(
        tensor,
        tensor,
        ctx,
        rank_global,
        dst_iris_rank,
        config,
    )
    ctx.device_barrier(group)


def recv(tensor, ctx, src, group=None, tag=0, config=None):
    """
    Receive tensor from source rank into tensor.

    The sender writes data directly into this tensor via iris.store.
    This function just waits for the barrier to ensure the data has arrived.

    Args:
        tensor: Output tensor to receive into, shape (M, N). Must be on
                the symmetric heap.
        ctx: Iris instance.
        src: Source rank within the group.
        group: ProcessGroup or None.
        tag: Communication tag (default: 0). Currently unused.
        config: Config with kernel parameters (default: None).
    """
    ctx.device_barrier(group)


def sendrecv(send_tensor, recv_tensor, ctx, dst, src, group=None, tag=0, config=None):
    """
    Simultaneous send and recv.

    Each rank writes its send_tensor to the destination rank's recv_tensor
    via iris.store, then a host-side barrier ensures all writes are visible.

    Args:
        send_tensor: Tensor to send, shape (M, N).
        recv_tensor: Tensor to receive into, shape (M, N). Must be on
                     the symmetric heap.
        ctx: Iris instance.
        dst: Destination rank for send.
        src: Source rank for recv.
        group: ProcessGroup or None.
        tag: Communication tag (default: 0). Currently unused.
        config: Config with kernel parameters (default: None).
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if dst < 0 or dst >= world_size:
        raise ValueError(f"dst rank {dst} is out of range for world_size {world_size}")
    if src < 0 or src >= world_size:
        raise ValueError(f"src rank {src} is out of range for world_size {world_size}")

    dst_iris_rank = rank_start + dst * rank_stride

    from iris.ccl.triton.sendrecv import launch_send

    launch_send(
        send_tensor,
        recv_tensor,
        ctx,
        rank_global,
        dst_iris_rank,
        config,
    )
    ctx.device_barrier(group)
