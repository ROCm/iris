# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import iris
from utils import read_realtime


@triton.jit
def tile_id_to_index_range(
    tile_id,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    tile_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m

    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N

    # clamp to the maximum valid index (M-1, N-1)
    max_m = M - 1
    max_n = N - 1

    # generate indices
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)

    rm = tl.minimum(rm, max_m)
    rn = tl.minimum(rn, max_n)

    return rm, rn, rm_start, rn_start


@triton.jit
def offset_for_tile(local_tile_id, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M_local, N_local):
    rm, rn, rm_start, rn_start = tile_id_to_index_range(
        local_tile_id, M_local, N_local, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    c_mask = (rm[:, None] < M_local) & (rn[None, :] < N_local)
    return rm, rn, c_mask, rm_start, rn_start


@triton.jit
def extract_submask_and_offset(
    rm,
    rn,
    mask,
    rm_start,
    rn_start,
    start_row,
    start_col,
    SUB_BLOCK_SIZE_M: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    stride_cm_local: tl.constexpr,
    stride_cn_local: tl.constexpr,
):
    # Create indices for the sub-block
    sub_rm = tl.arange(0, SUB_BLOCK_SIZE_M) + start_row
    sub_rn = tl.arange(0, SUB_BLOCK_SIZE_N) + start_col

    # Create a 2D grid of indices for the sub-block
    sub_rm_2d = sub_rm[:, None]  # Shape: (SUB_BLOCK_SIZE_M, 1)
    sub_rn_2d = sub_rn[None, :]  # Shape: (1, SUB_BLOCK_SIZE_N)

    # Compute the sub-mask
    sub_mask = (sub_rm_2d < BLOCK_SIZE_M) & (sub_rn_2d < BLOCK_SIZE_N)

    # Compute the sub-offset relative to the start of the tile
    sub_offset = ((rm_start + sub_rm_2d) * stride_cm_local) + ((rn_start + sub_rn_2d) * stride_cn_local)

    return sub_mask, sub_offset


@triton.jit
def all_reduce_kernel(
    partial_c,
    c,
    tile_completed_ptr,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    total_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCTION_TILE_M: tl.constexpr = 128,
    REDUCTION_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(axis=0)

    for tile in range(pid, total_tiles, NUM_SMS):
        result = 0

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

        while result == 0:
            compare = 1
            value = 0
            result = iris.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile in sub-tiles
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M_local, N_local
        )

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, REDUCTION_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, REDUCTION_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        # Flattened loop over all sub-tiles, triton is
        # better at handling flat loops instead of nested loops
        for sub_tile_idx in range(0, total_sub_tiles):
            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * REDUCTION_TILE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * REDUCTION_TILE_N

            # Extract the sub-mask and sub-offset for the current sub-block
            sub_mask, sub_offset = extract_submask_and_offset(
                rm,
                rn,
                mask,
                rm_start,
                rn_start,
                start_row,
                start_col,
                REDUCTION_TILE_M,
                REDUCTION_TILE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                stride_cm_local,
                stride_cn_local,
            )

            # Load data from the local partial result
            data = tl.load(partial_c + sub_offset, mask=sub_mask)

            # Store data to the global result using atomic_add
            for remote_rank in range(world_size):
                iris.atomic_add(
                    c + sub_offset,
                    data,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=sub_mask,
                    scope="sys",
                )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(end_timestamp_ptr + tile, timestamp)


@triton.jit
def all_scatter_kernel(
    partial_c,
    c,
    tile_completed_ptr,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M_global: tl.constexpr,
    total_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    SCATTER_TILE_M: tl.constexpr = 128,
    SCATTER_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(axis=0)
    pid = (pid % 8) * (NUM_SMS // 8) + (pid // 8)

    for tile in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

        result = 0
        # Spin till tile is produced
        while result == 0:
            compare = 1
            value = 0
            result = iris.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M_global, M_local, N_local
        )

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, SCATTER_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, SCATTER_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        # Flattened loop over all sub-tiles, triton is
        # better at handling flat loops instead of nested loops
        for sub_tile_idx in range(0, total_sub_tiles):
            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * SCATTER_TILE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * SCATTER_TILE_N

            # Extract the sub-mask and sub-offset for the current sub-block
            sub_mask, sub_offset = extract_submask_and_offset(
                rm,
                rn,
                mask,
                rm_start,
                rn_start,
                start_row,
                start_col,
                SCATTER_TILE_M,
                SCATTER_TILE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                stride_cm_local,
                stride_cn_local,
            )

            # Load data from the local partial result
            data = tl.load(partial_c + sub_offset, mask=sub_mask, cache_modifier=".cv")

            # Translate to global
            sub_mask, global_offset = extract_submask_and_offset(
                rm,
                rn + cur_rank * N_local,
                mask,
                rm_start,
                rn_start + cur_rank * N_local,
                start_row,
                start_col,
                SCATTER_TILE_M,
                SCATTER_TILE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                stride_cm_global,
                stride_cn_global,
            )

            # Store data to the global result using puts
            for remote_rank in range(world_size):
                if remote_rank == cur_rank:
                    # For the current rank, we can use store
                    tl.store(c + global_offset, data, mask=sub_mask)
                else:
                    iris.put(
                        c + global_offset,
                        data,
                        cur_rank,
                        remote_rank,
                        heap_bases,
                        mask=sub_mask,
                    )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(end_timestamp_ptr + tile, timestamp)


@triton.jit
def all_gather_kernel(
    partial_c,
    c,
    tile_completed_ptr,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M_global: tl.constexpr,
    total_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    GATHER_TILE_M: tl.constexpr = 128,
    GATHER_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(axis=0)
    pid = (pid % 8) * (NUM_SMS // 8) + (pid // 8)

    for tile in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

        # Spin untill all ranks have produced the tile.
        result = 0
        while result < world_size:
            compare = world_size
            value = 0
            result = iris.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M_global, M_local, N_local
        )

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, GATHER_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, GATHER_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        for remote_rank in range(world_size):
            # Flattened loop over all sub-tiles, triton is
            # better at handling flat loops instead of nested loops
            for sub_tile_idx in range(0, total_sub_tiles):
                # Calculate start_row and start_col for the current sub-tile
                start_row = (sub_tile_idx // num_sub_tiles_n) * GATHER_TILE_M
                start_col = (sub_tile_idx % num_sub_tiles_n) * GATHER_TILE_N

                # Extract the sub-mask and sub-offset for the current sub-block
                sub_mask, sub_offset = extract_submask_and_offset(
                    rm,
                    rn,
                    mask,
                    rm_start,
                    rn_start,
                    start_row,
                    start_col,
                    GATHER_TILE_M,
                    GATHER_TILE_N,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    stride_cm_local,
                    stride_cn_local,
                )

                # Load data from the local partial result.
                if remote_rank == cur_rank:
                    data = tl.load(partial_c + sub_offset, mask=sub_mask, cache_modifier=".cv")
                # For other ranks, we need to get the data from the remote rank.
                else:
                    data = iris.get(
                        partial_c + sub_offset,
                        cur_rank,
                        remote_rank,
                        heap_bases,
                        mask=sub_mask,
                    )

                # Translate to global
                sub_global_mask, global_offset = extract_submask_and_offset(
                    rm,
                    rn + remote_rank * N_local,
                    mask,
                    rm_start,
                    rn_start + remote_rank * N_local,
                    start_row,
                    start_col,
                    GATHER_TILE_M,
                    GATHER_TILE_N,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    stride_cm_global,
                    stride_cn_global,
                )

                # Store data to the global result using store
                tl.store(c + global_offset, data, mask=sub_global_mask)

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(end_timestamp_ptr + tile, timestamp)


@triton.jit
def one_shot_kernel(
    partial_c,
    c,
    tile_completed_ptr,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    total_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCTION_TILE_M: tl.constexpr = 128,
    REDUCTION_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(axis=0)
    pid = (pid % 8) * (NUM_SMS // 8) + (pid // 8)

    for tile in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

        result = 0
        # Wait for first 2 ranks to produce this tile
        # TODO: This is a HACK!
        while result < 2:  # world_size:
            compare = world_size
            value = 0
            result = iris.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile in sub-tiles
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, M_local, N_local
        )

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, REDUCTION_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, REDUCTION_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        # Flattened loop over all sub-tiles, triton is
        # better at handling flat loops instead of nested loops
        for sub_tile_idx in range(0, total_sub_tiles):
            acc = tl.zeros((REDUCTION_TILE_M, REDUCTION_TILE_N), dtype=tl.float32)

            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * REDUCTION_TILE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * REDUCTION_TILE_N

            # Extract the sub-mask and sub-offset for the current sub-block
            sub_mask, sub_offset = extract_submask_and_offset(
                rm,
                rn,
                mask,
                rm_start,
                rn_start,
                start_row,
                start_col,
                REDUCTION_TILE_M,
                REDUCTION_TILE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                stride_cm_local,
                stride_cn_local,
            )

            # Simpler logic for reduction:
            # for remote_rank in range(0, world_size):
            #     acc += iris.get(
            #         partial_c + sub_offset,
            #         cur_rank,
            #         remote_rank,
            #         heap_bases,
            #         mask=sub_mask,
            #     )

            # Advanced buffer swapping logic:
            # For a given sub-tile, we will prefetch the data into rr first
            # and while that is being reduced into acc, we will prefetch the next
            # sub-tile into pp. This way we can overlap the reduction with the
            # prefetching of the next sub-tile.
            rr = iris.get(
                partial_c + sub_offset,
                cur_rank,
                0,
                heap_bases,
                mask=sub_mask,
            )

            pp = tl.zeros((REDUCTION_TILE_M, REDUCTION_TILE_N), dtype=tl.float16)
            tl.debug_barrier()  # Ensure rr is ready before proceeding.

            for remote_rank in range(1, world_size):
                acc += rr
                pp = iris.get(
                    partial_c + sub_offset,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=sub_mask,
                )
                tl.debug_barrier()  # Ensure pp is ready before proceeding.

                # Swap buffers:
                tmp = pp
                pp = rr
                rr = tmp

            # For the last rank, we need to add the accumulated value
            acc += rr

            # Sub-tile completed, store the output as a sub-tile to c
            tl.store(c + sub_offset, acc, mask=sub_mask, cache_modifier=".wt")

            if COLLECT_TIMESTAMPS:
                timestamp = read_realtime()
                tl.atomic_max(end_timestamp_ptr + tile, timestamp)
