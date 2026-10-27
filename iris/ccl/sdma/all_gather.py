# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather on the SDMA copy engines.

``launch`` backs ``all_gather(config=Config(engine="sdma"))``: a kernel queues each rank's slice
to every peer on the device-initiated copy-engine queues. ``prepare`` backs
``prepare_all_gather``: a triggered plan whose copies the producer kernel releases.
"""

import numpy as np
import triton
import triton.language as tl

import iris
from iris.device import sdma_utils
from iris.host.distributed.helpers import distributed_allgather
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.mem.triton.context import __translate

# COPY_LINEAR's count field is 30 bits on SDMA 4.4.x
MAX_COPY_BYTES = 1 << 30


@triton.jit
def _copy(copy_engine_ctx, to_rank, src, dst, nbytes):
    """One COPY_LINEAR of ``nbytes`` (at most 1 GiB) on the device-initiated queue to ``to_rank``."""
    handle = copy_engine_ctx + sdma_utils.sdma_ep.QUEUE_DEVICE_CTX_SIZE * to_rank
    queue_ptr_u32 = tl.load(handle + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(handle + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(handle + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(handle + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(handle + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(handle + 5).to(tl.pointer_type(tl.uint64))

    command_in_bytes = sdma_utils.sdma_ep.COPY_LINEAR_COMMAND_BYTES
    base, offset = sdma_utils.acquire_fadd(
        queue_ptr_u32,
        read_ptr,
        write_ptr,
        doorbell_ptr,
        cached_write_ptr,
        committed_write_ptr,
        command_in_bytes,
    )
    sdma_utils.place_nop_packet(queue_ptr_u32, base, offset)
    sdma_utils.place_copy_packet(queue_ptr_u32, base + offset, nbytes.to(tl.uint32), src, dst)
    sdma_utils.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, base + offset + command_in_bytes)


@triton.jit
def sdma_all_gather(
    input_ptr,
    slice_ptr,
    nbytes,
    counters,
    expected,
    heap_bases,
    copy_engine_ctx,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    MAX_COPY: tl.constexpr,
):
    """
    Program ``i`` sends this rank's slice to its ``i``-th peer and adds 1 to this rank's counter
    there, then waits until the peer's slice has arrived here and this rank's has arrived there.
    """
    i = tl.program_id(0)
    peer = rank_start + (i + (i >= group_rank).to(tl.int32)) * rank_stride

    src = input_ptr.to(tl.uint64)
    dst = __translate(slice_ptr, iris_rank, peer, heap_bases).to(tl.uint64)
    for start in range(0, nbytes, MAX_COPY):
        _copy(copy_engine_ctx, peer, src + start, dst + start, tl.minimum(nbytes - start, MAX_COPY))
    # Runs after the copies on the same queue
    iris.atomic_add(
        counters + iris_rank, 1, iris_rank, peer, heap_bases, copy_engine_ctx=copy_engine_ctx, use_copy_engine=True
    )

    target = tl.load(expected + peer) + 1
    while tl.atomic_add(counters + peer, 0, sem="acquire", scope="sys") < target:
        pass
    # This rank's copies to the peer are done, so input may be reused
    while iris.atomic_add(counters + iris_rank, 0, iris_rank, peer, heap_bases, sem="acquire", scope="sys") < target:
        pass
    tl.store(expected + peer, target)


def launch(input_tensor, output_tensor, ctx, rank_in_group, rank_global, world_size, rank_start, rank_stride):
    """Copy this rank's slice into ``output_tensor``, send it to every peer, and wait for theirs."""
    ctx._require_copy_engine()
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("engine='sdma' needs contiguous input and output tensors")
    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(f"input dtype {input_tensor.dtype} does not match output dtype {output_tensor.dtype}")
    if not ctx.heap.is_symmetric(output_tensor):
        raise ValueError("engine='sdma' needs the output tensor on the symmetric heap")

    n = input_tensor.numel()
    mine = output_tensor.view(-1)[rank_in_group * n : (rank_in_group + 1) * n]
    if mine.data_ptr() != input_tensor.data_ptr():
        mine.copy_(input_tensor.view(-1))
    if world_size == 1:
        return

    # Counters only grow, by one per call between two ranks, so every group can share them
    counters, expected = ctx._sdma_all_gather_counters
    iris_launch(
        sdma_all_gather,
        (world_size - 1,),
        input_tensor,
        mine,
        n * input_tensor.element_size(),
        counters,
        expected,
        ctx.get_heap_bases(),
        ctx.get_copy_engine_ctx(),
        rank_in_group,
        rank_global,
        rank_start,
        rank_stride,
        MAX_COPY_BYTES,
        num_warps=1,
        algorithm="all_gather",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )


def prepare(output_tensor, ctx, schedule=None, timeout=30.0):
    """Collective: a triggered plan that gathers ``output_tensor`` in place. See ``CCL.prepare_all_gather``."""
    from iris.experimental.plan import TriggeredPlan, whole

    rank, world_size = ctx.get_rank(), ctx.get_num_ranks()
    error = None
    try:
        if not output_tensor.is_contiguous() or output_tensor.dim() == 0:
            raise ValueError("output_tensor must be contiguous with at least one dimension")
        if output_tensor.shape[0] % world_size:
            raise ValueError(
                f"output_tensor's first dimension {output_tensor.shape[0]} is not a multiple of {world_size}"
            )
        rows = output_tensor.shape[0] // world_size
        mine = output_tensor[rank * rows : (rank + 1) * rows]
        if schedule is None:
            schedule = whole(mine, base=output_tensor)
        slice_bytes = mine.numel() * mine.element_size()
        lo, hi = rank * slice_bytes, (rank + 1) * slice_bytes
        for i, chunk in enumerate(schedule):
            for region in chunk.regions:
                if region.offset_bytes < lo or region.end > hi:
                    raise ValueError(
                        f"chunk {i}: region {region} lies outside rank {rank}'s slice (bytes {lo} to {hi}); "
                        f"build the schedule from output_tensor[{rank * rows}:{(rank + 1) * rows}]"
                    )
    except Exception as e:
        error = e
    # Agree before the plan's own collectives, so a rank that raised leaves none waiting
    failed = distributed_allgather(np.array([error is not None], dtype=np.int64))[:, 0].nonzero()[0].tolist()
    if error is not None:
        raise error
    if failed:
        raise RuntimeError(f"rank(s) {failed} could not prepare the all-gather")
    return TriggeredPlan(ctx, output_tensor, schedule, timeout=timeout)
