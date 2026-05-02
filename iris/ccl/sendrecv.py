# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Point-to-point send/recv operations — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

import torch
from iris.ccl.utils import extract_group_info

# Maximum number of distinct tags supported concurrently.  Tags wrap
# modulo this value, so callers using more than _MAX_SENDRECV_TAGS
# distinct tags concurrently risk flag collisions.
_MAX_SENDRECV_TAGS = 16


def _get_sendrecv_flags(ctx):
    """
    Allocate the per-context flags tensor for send/recv synchronization.

    Returns a flat int32 tensor of size (world_size * _MAX_SENDRECV_TAGS)
    on the symmetric heap.  Every rank allocates this tensor at the same
    point in the heap allocation sequence, so the heap offset is identical
    across all ranks.  That symmetry is critical: the sender writes a flag
    on the *receiver's* heap via iris RMA, and both sides must agree on
    the address.

    The tensor is indexed as flags[src_rank * _MAX_SENDRECV_TAGS + tag].
    A flag at [s, t] on rank R's heap means "rank s has finished sending
    to rank R for tag t".

    Flags are stored as an attribute on the context object itself so they
    are automatically cleaned up when the context is destroyed. This avoids
    stale cache entries from the previous mutable-default-argument approach,
    which caused SIGABRT crashes when tests created multiple iris instances.
    """
    if not hasattr(ctx, "_sendrecv_flags"):
        world_size = ctx.get_num_ranks()
        ctx._sendrecv_flags = ctx.zeros((world_size * _MAX_SENDRECV_TAGS,), dtype=torch.int32)
    return ctx._sendrecv_flags


def _get_sendrecv_flag(ctx, src_rank, tag):
    """
    Return a 1-element view into the flags tensor for a specific
    (src_rank, tag) pair.

    Both sender and receiver call this with the *same* src_rank (the
    global iris rank of the sending side) and tag, so they get a view at
    the same heap offset.  The flag lives on the receiver's heap; the
    sender writes to it via iris.atomic_xchg remotely.
    """
    flags = _get_sendrecv_flags(ctx)
    idx = src_rank * _MAX_SENDRECV_TAGS + (tag % _MAX_SENDRECV_TAGS)
    return flags[idx : idx + 1]


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

    # Flag is keyed by (src_rank=rank_global, tag) so both sender and
    # receiver resolve to the same heap offset.  The flag lives on the
    # receiver's heap and is written remotely by the sender.
    flag = _get_sendrecv_flag(ctx, src_rank=rank_global, tag=tag)

    if config.use_gluon:
        from iris.ccl.gluon.sendrecv import launch_send
    else:
        from iris.ccl.triton.sendrecv import launch_send

    launch_send(
        tensor,
        tensor,  # output_ptr = sender's tensor, same heap offset as receiver's buffer
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

    # Flag is keyed by (src_rank=src_iris_rank, tag) — same key the
    # sender used, so both sides point to the same heap offset.
    # The flag is on OUR heap, written by the sender via iris RMA.
    flag = _get_sendrecv_flag(ctx, src_rank=src_iris_rank, tag=tag)

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

    # send_flag: keyed by (src=rank_global, tag) — we are the sender,
    # so src_rank is our own global rank.  The flag lives on dst's heap.
    send_flag = _get_sendrecv_flag(ctx, src_rank=rank_global, tag=tag)

    # recv_flag: keyed by (src=src_iris_rank, tag) — the remote sender's
    # rank.  The flag lives on OUR heap.
    recv_flag = _get_sendrecv_flag(ctx, src_rank=src_iris_rank, tag=tag)

    if config.use_gluon:
        from iris.ccl.gluon.sendrecv import launch_send, launch_recv
    else:
        from iris.ccl.triton.sendrecv import launch_send, launch_recv

    # Launch send — output_ptr must be recv_tensor (same heap offset on all
    # ranks) so iris.store lands data at the receiver's recv_tensor address.
    # Using send_tensor here would write to the wrong heap offset on the
    # destination because send_tensor and recv_tensor are at different offsets.
    launch_send(
        send_tensor,
        recv_tensor,
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
