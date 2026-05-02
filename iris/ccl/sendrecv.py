# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Point-to-point send/recv operations — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

import torch
from iris.ccl.utils import extract_group_info


def _get_sendrecv_flag(ctx, peer_rank, tag, _cache={}):
    """
    Get or create a flag tensor for send/recv synchronization.

    Each (peer_rank, tag) pair gets a unique flag. The flag is allocated
    on the symmetric heap so it is accessible from all ranks.

    Returns:
        int32 tensor with one element, initialized to 0.
    """
    key = (id(ctx), peer_rank, tag)
    if key not in _cache:
        _cache[key] = ctx.zeros((1,), dtype=torch.int32)
    return _cache[key]


def send(tensor, ctx, dst, group=None, tag=0, config=None):
    """
    Send tensor to destination rank.

    The sender writes data directly to the receiver's output buffer via
    XGMI/P2P remote writes, then signals completion with an atomic flag.

    Args:
        tensor: Tensor to send, shape (M, N).
        ctx: Iris instance.
        dst: Destination rank within the group.
        group: ProcessGroup or None. If None, uses all ranks in ctx.
        tag: Communication tag for matching send/recv pairs (default: 0).
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

    # Compute the global iris rank of the destination
    dst_iris_rank = rank_start + dst * rank_stride

    # The receiver's output buffer is at the same heap offset as our tensor.
    # We write into the receiver's copy of this tensor via iris.store.
    # The flag tells the receiver when the write is complete.
    flag = _get_sendrecv_flag(ctx, dst_iris_rank, tag)

    if config.use_gluon:
        from iris.ccl.gluon.sendrecv import launch_send
    else:
        from iris.ccl.triton.sendrecv import launch_send

    launch_send(
        tensor,
        tensor,  # output_ptr = same tensor, different rank's address space
        flag,
        ctx,
        rank_global,
        dst_iris_rank,
        tag,
        config,
    )


def recv(tensor, ctx, src, group=None, tag=0, config=None):
    """
    Receive tensor from source rank into tensor.

    The receiver spins waiting for the sender's completion flag. Once
    signaled, data is already in place in tensor (written by sender's
    iris.store).

    Args:
        tensor: Output tensor to receive into, shape (M, N). Must be on
                the symmetric heap (allocated via ctx.zeros or similar).
        ctx: Iris instance.
        src: Source rank within the group.
        group: ProcessGroup or None. If None, uses all ranks in ctx.
        tag: Communication tag for matching send/recv pairs (default: 0).
        config: Config with kernel parameters (default: None).
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if src < 0 or src >= world_size:
        raise ValueError(f"src rank {src} is out of range for world_size {world_size}")
    if src == rank_in_group:
        raise ValueError("Cannot recv from self — use tensor.copy_() instead")

    # Compute the global iris rank of the source
    src_iris_rank = rank_start + src * rank_stride

    # The flag is on OUR heap, written by the sender via iris.store
    flag = _get_sendrecv_flag(ctx, src_iris_rank, tag)

    if config.use_gluon:
        from iris.ccl.gluon.sendrecv import launch_recv
    else:
        from iris.ccl.triton.sendrecv import launch_recv

    launch_recv(
        flag,
        rank_global,
        config,
    )


def sendrecv(send_tensor, recv_tensor, ctx, dst, src, group=None, tag=0, config=None):
    """
    Simultaneous send and recv.

    Sends send_tensor to dst rank while receiving into recv_tensor from
    src rank. Both operations are launched, then a barrier ensures
    completion.

    Args:
        send_tensor: Tensor to send, shape (M, N).
        recv_tensor: Tensor to receive into, shape (M, N). Must be on
                     the symmetric heap.
        ctx: Iris instance.
        dst: Destination rank for send.
        src: Source rank for recv.
        group: ProcessGroup or None. If None, uses all ranks in ctx.
        tag: Communication tag (default: 0). Same tag used for both.
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
    src_iris_rank = rank_start + src * rank_stride

    send_flag = _get_sendrecv_flag(ctx, dst_iris_rank, tag)
    recv_flag = _get_sendrecv_flag(ctx, src_iris_rank, tag)

    if config.use_gluon:
        from iris.ccl.gluon.sendrecv import launch_send, launch_recv
    else:
        from iris.ccl.triton.sendrecv import launch_send, launch_recv

    # Launch send
    launch_send(
        send_tensor,
        send_tensor,
        send_flag,
        ctx,
        rank_global,
        dst_iris_rank,
        tag,
        config,
    )

    # Launch recv (waits for flag from src)
    launch_recv(
        recv_flag,
        rank_global,
        config,
    )

    ctx.barrier()
