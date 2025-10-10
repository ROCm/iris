# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

import iris


@triton.jit()
def persistent_all_reduce(
    partials,
    ring_buffer,
    output,
    flags,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (COMM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Ring topology
    next_rank = (cur_rank + 1) % world_size
    prev_rank = (cur_rank + world_size - 1) % world_size

    acc_dtype = tl.float32 if output.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Begin: See the if segment for explanation:
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        offset = rm[:, None] * stride_cm + rn[None, :] * stride_cn
        # End: masks/offset calculations.

        # Initialize accumulator with local partial result from ring_buffer
        acc = tl.load(partials + offset, mask=mask).to(acc_dtype)

        # Each rank sends its LOCAL partial result (not accumulated) around the ring
        # while accumulating received partial results from other ranks.
        #
        # Initial: Each rank has computed a partial-K GEMM result in 'acc'
        # Goal: Sum all partial results from all ranks
        #
        # Algorithm: Use ring_buffer to pass data around, accumulate locally
        # - send_data: What we send (starts as our partial result)
        # - acc: Running sum of all partial results received so far

        # Initialize: First, write our partial result to ring_buffer for sending
        send_data = acc

        # Step loop: send to next, wait/recv from prev, add.
        for _step in range(0, world_size - 1):
            # 1a) Wait for NEXT rank to be ready (its lock should be 0, meaning it finished previous step)
            # This prevents overwriting data that hasn't been consumed yet
            while (
                iris.atomic_cas(flags + tile_id, 0, 0, cur_rank, next_rank, heap_bases, sem="acquire", scope="sys") != 0
            ):
                pass

            # 1b) Send our current accumulator tile to NEXT rank's ring buffer
            iris.store(ring_buffer + offset, send_data, cur_rank, next_rank, heap_bases, mask=mask)

            tl.debug_barrier()
            # Signal "ready" by setting NEXT rank's flag for this tile to 1
            iris.atomic_xchg(flags + tile_id, 1, cur_rank, next_rank, heap_bases, sem="release", scope="sys")

            # 2) Wait for PREV rank to signal our local flag for this tile
            while tl.atomic_cas(flags + tile_id, 0, 0, sem="acquire", scope="sys") != 1:
                pass

            # 3) Consume the received tile from our LOCAL ring buffer (prev wrote here)
            recv_tile = tl.load(ring_buffer + offset, mask=mask, other=tl.zeros_like(acc))

            # 4) Accumulate received data and prepare to forward it in next iteration
            acc += recv_tile  # tl.load(ring_buffer + offset, mask=mask)
            send_data = recv_tile  # Forward what we just received (not the accumulated sum)

            # 5) Reset our local flag to 0 (done consuming this step)
            tl.atomic_xchg(flags + tile_id, 0, sem="release", scope="sys")

        # Write fully-reduced tile to local result buffer (no remote writes)
        o = acc.to(output.type.element_ty)
        tl.store(output + offset, o, mask=mask)
