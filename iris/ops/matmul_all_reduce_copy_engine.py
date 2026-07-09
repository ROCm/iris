# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
High-level API for fused matrix multiplication and all-reduce.

This module provides a torch-like interface for GEMM+All-Reduce operations,
automatically inferring dimensions, strides, and hardware parameters.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
from xio import sdma_ep

from tritonblas.kernels.stages import GemmContext, Tile, make_input_view
from tritonblas.matmul import _make_matmul_selector

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from .tritonblas_launch_wave_schedule import build_launch_wave_plan


@triton.jit()
def _partitioned_xcd_matmul_kernel(
    A,
    B,
    local_aux_buffer,
    reduce_buffer,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_local_aux_m,
    stride_local_aux_n,
    stride_reduce_m,
    stride_reduce_n,
    cur_rank: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    OUTPUT_PARTITIONS: tl.constexpr,
    MAX_BATCHES_PER_PARTITION: tl.constexpr,
    STORE_LOCAL_REDUCE_SHARD: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Persistent GEMM that pins output row partitions to XCD-local work.

    Program ids are interpreted as interleaved XCD lanes: pid % NUM_XCDS is the
    XCD lane, and each output row partition is assigned to partition % NUM_XCDS.
    On MI300 this keeps each rank/partition shard local to one XCD while the
    GEMM epilogue publishes batch flags for SDMA.
    """
    acc_dtype = tl.int32 if local_aux_buffer.type.element_ty == tl.int8 else tl.float32
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        chunk_size=CHUNK_SIZE,
        cache_modifier_a=None,
        cache_modifier_b=None,
        acc_dtype=acc_dtype,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)

    launch_pid = tl.program_id(0)
    xcd_id = launch_pid % NUM_XCDS
    local_pid = launch_pid // NUM_XCDS
    sms_per_xcd = NUM_SMS // NUM_XCDS

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    partition_rows = tl.cdiv(M, OUTPUT_PARTITIONS)

    for partition_id in range(OUTPUT_PARTITIONS):
        if partition_id % NUM_XCDS == xcd_id:
            partition_m_start = partition_id * partition_rows
            partition_m_end = tl.minimum(partition_m_start + partition_rows, M)
            first_pid_m = partition_m_start // BLOCK_SIZE_M
            last_pid_m = tl.cdiv(partition_m_end, BLOCK_SIZE_M)
            partition_tiles_m = last_pid_m - first_pid_m
            partition_tiles = partition_tiles_m * num_pid_n

            for local_tile_id in range(local_pid, partition_tiles, sms_per_xcd):
                batch_iter = local_tile_id // sms_per_xcd
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = local_tile_id // num_pid_in_group
                first_group_pid_m = group_id * GROUP_SIZE_M
                group_size_m = tl.minimum(partition_tiles_m - first_group_pid_m, GROUP_SIZE_M)

                local_pid_m = first_group_pid_m + ((local_tile_id % num_pid_in_group) % group_size_m)
                pid_m = first_pid_m + local_pid_m
                pid_n = (local_tile_id % num_pid_in_group) // group_size_m

                tl.assume(pid_m >= 0)
                tl.assume(pid_n >= 0)

                out_tile = Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
                acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)
                rm, rn = out_tile.indices()
                mask = (
                    (rm[:, None] >= partition_m_start)
                    & (rm[:, None] < partition_m_end)
                    & (rm[:, None] < M)
                    & (rn[None, :] < N)
                )
                c = acc.to(local_aux_buffer.type.element_ty)

                if STORE_LOCAL_REDUCE_SHARD and partition_id == cur_rank:
                    rows_per_rank = M // OUTPUT_PARTITIONS
                    local_rm = rm - partition_m_start
                    owner_row_mask = (rm >= partition_m_start) & (rm < partition_m_end)
                    staged_rm = cur_rank * rows_per_rank + tl.where(owner_row_mask, local_rm, 0)
                    reduce_ptr = reduce_buffer + staged_rm[:, None] * stride_reduce_m + rn[None, :] * stride_reduce_n
                    tl.store(reduce_ptr, c, mask=mask & owner_row_mask[:, None], cache_modifier=".wt")
                else:
                    aux_ptr = local_aux_buffer + rm[:, None] * stride_local_aux_m + rn[None, :] * stride_local_aux_n
                    tl.store(aux_ptr, c, mask=mask, cache_modifier=".wt")

                tl.debug_barrier()
                batch_id = partition_id * MAX_BATCHES_PER_PARTITION + batch_iter
                tl.atomic_add(locks + batch_id, 1, sem="release", scope="sys")


@triton.jit()
def _matmul_all_reduce_copy_engine_wait_completion_kernel(
    completion_signals,
    expected_value,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    src_rank = tl.program_id(0)
    if src_rank >= world_size or src_rank == cur_rank:
        return
    while tl.load(completion_signals + src_rank, cache_modifier=".cv", volatile=True) < expected_value:
        pass


@triton.jit()
def _matmul_all_reduce_copy_engine_reduce_scatter_kernel(
    C,
    reduce_buffer,
    completion_signals,
    expected_completion_value,
    M,
    N,
    stride_reduce_m,
    stride_reduce_n,
    stride_cm,
    stride_cn,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WAIT_FOR_COMPLETION: tl.constexpr,
):
    """
    Persistent reduce-scatter kernel with CCL-style tile distribution.

    Each workgroup processes multiple tiles to reduce scheduling overhead.
    Reduces this rank's row band, then all-gathers it to every rank.
    """
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.make_tensor_view(C, M, N, stride_cm, stride_cn)

    if WAIT_FOR_COMPLETION:
        for wait_src_rank in tl.static_range(0, world_size):
            if wait_src_rank != cur_rank:
                while tl.load(
                    completion_signals + wait_src_rank,
                    cache_modifier=".cv",
                    volatile=True,
                ) < expected_completion_value:
                    pass

    pid = tl.program_id(0)
    rows_per_rank = M // world_size
    num_pid_m = tl.cdiv(rows_per_rank, REDUCE_BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, REDUCE_BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # Persistent loop with CCL-style tile distribution
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # CCL-style swizzled tile coordinates
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        local_pid_m = pid_m

        rm_base = local_pid_m * REDUCE_BLOCK_SIZE_M
        rm = rm_base + tl.arange(0, REDUCE_BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, REDUCE_BLOCK_SIZE_M), REDUCE_BLOCK_SIZE_M)

        rn_base = pid_n * REDUCE_BLOCK_SIZE_N
        rn = rn_base + tl.arange(0, REDUCE_BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, REDUCE_BLOCK_SIZE_N), REDUCE_BLOCK_SIZE_N)

        mask = (rm[:, None] < rows_per_rank) & (rn[None, :] < N)

        acc = tl.zeros((REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N), dtype=acc_dtype)
        for reduce_src_rank in range(world_size):
            src_rm = reduce_src_rank * rows_per_rank + rm
            src_ptr = reduce_buffer + src_rm[:, None] * stride_reduce_m + rn[None, :] * stride_reduce_n
            data = tl.load(src_ptr, mask=mask, other=0.0)
            acc += data.to(acc_dtype)

        tile_obj = iris.Tile(
            local_pid_m,
            pid_n,
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            acc.to(C.type.element_ty),
        )
        ctx.all_gather(tile_obj, dst_view, dim=0)


@triton.jit()
def _matmul_all_reduce_copy_engine_local_reduce_kernel(
    C,
    local_aux_buffer,
    remote_inbox,
    completion_signals,
    expected_completion_value,
    M,
    N,
    stride_local_aux_m,
    stride_local_aux_n,
    stride_remote_m,
    stride_remote_n,
    stride_cm,
    stride_cn,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WAIT_FOR_COMPLETION: tl.constexpr,
):
    """Local one-shot reduction after SDMA has gathered every peer's full partial output."""
    if WAIT_FOR_COMPLETION:
        for wait_src_rank in tl.static_range(0, world_size):
            if wait_src_rank != cur_rank:
                while tl.load(
                    completion_signals + wait_src_rank,
                    cache_modifier=".cv",
                    volatile=True,
                ) < expected_completion_value:
                    pass

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, REDUCE_BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, REDUCE_BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * REDUCE_BLOCK_SIZE_M
        rn_base = pid_n * REDUCE_BLOCK_SIZE_N
        is_full = (rm_base + REDUCE_BLOCK_SIZE_M <= M) & (rn_base + REDUCE_BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, REDUCE_BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, REDUCE_BLOCK_SIZE_M), REDUCE_BLOCK_SIZE_M)

        rn = rn_base + tl.arange(0, REDUCE_BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, REDUCE_BLOCK_SIZE_N), REDUCE_BLOCK_SIZE_N)

        local_ptr = local_aux_buffer + rm[:, None] * stride_local_aux_m + rn[None, :] * stride_local_aux_n
        out_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn

        if is_full:
            fast_local_ptr = tl.max_contiguous(
                tl.multiple_of(local_ptr, (1, REDUCE_BLOCK_SIZE_N)),
                (1, REDUCE_BLOCK_SIZE_N),
            )
            acc = tl.load(fast_local_ptr).to(acc_dtype)
            for reduce_src_rank in tl.static_range(0, world_size):
                if reduce_src_rank != cur_rank:
                    src_rm = reduce_src_rank * M + rm
                    src_ptr = remote_inbox + src_rm[:, None] * stride_remote_m + rn[None, :] * stride_remote_n
                    fast_src_ptr = tl.max_contiguous(
                        tl.multiple_of(src_ptr, (1, REDUCE_BLOCK_SIZE_N)),
                        (1, REDUCE_BLOCK_SIZE_N),
                    )
                    data = tl.load(fast_src_ptr)
                    acc += data.to(acc_dtype)

            fast_out_ptr = tl.max_contiguous(
                tl.multiple_of(out_ptr, (1, REDUCE_BLOCK_SIZE_N)),
                (1, REDUCE_BLOCK_SIZE_N),
            )
            tl.store(fast_out_ptr, acc.to(C.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            acc = tl.load(local_ptr, mask=mask, other=0.0).to(acc_dtype)

            for reduce_src_rank in tl.static_range(0, world_size):
                if reduce_src_rank != cur_rank:
                    src_rm = reduce_src_rank * M + rm
                    src_ptr = remote_inbox + src_rm[:, None] * stride_remote_m + rn[None, :] * stride_remote_n
                    data = tl.load(src_ptr, mask=mask, other=0.0)
                    acc += data.to(acc_dtype)

            tl.store(out_ptr, acc.to(C.type.element_ty), mask=mask)


@triton.jit()
def _matmul_all_reduce_copy_engine_local_reduce_flat_kernel(
    C,
    local_aux_buffer,
    remote_inbox,
    completion_signals,
    expected_completion_value,
    total_elements,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WAIT_FOR_COMPLETION: tl.constexpr,
):
    """Fast one-shot local reduction for contiguous rank-major buffers."""
    if WAIT_FOR_COMPLETION:
        for wait_src_rank in tl.static_range(0, world_size):
            if wait_src_rank != cur_rank:
                while tl.load(
                    completion_signals + wait_src_rank,
                    cache_modifier=".cv",
                    volatile=True,
                ) < expected_completion_value:
                    pass

    pid = tl.program_id(0)
    total_blocks = total_elements // BLOCK_SIZE
    block_offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = tl.max_contiguous(tl.multiple_of(block_offsets, BLOCK_SIZE), BLOCK_SIZE)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    for block_id in range(pid, total_blocks, NUM_SMS):
        linear_base = block_id * BLOCK_SIZE
        linear_offsets = linear_base + block_offsets

        local_ptr = local_aux_buffer + linear_offsets
        local_ptr = tl.max_contiguous(tl.multiple_of(local_ptr, BLOCK_SIZE), BLOCK_SIZE)

        if world_size == 8:
            if cur_rank == 0:
                data0 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data0 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 1:
                data1 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data1 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 2:
                data2 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = 2 * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data2 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 3:
                data3 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = 3 * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data3 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 4:
                data4 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = 4 * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data4 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 5:
                data5 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = 5 * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data5 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 6:
                data6 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = 6 * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data6 = tl.load(src_ptr).to(acc_dtype)

            if cur_rank == 7:
                data7 = tl.load(local_ptr).to(acc_dtype)
            else:
                src_offsets = 7 * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data7 = tl.load(src_ptr).to(acc_dtype)

            acc = ((data0 + data1) + (data2 + data3)) + ((data4 + data5) + (data6 + data7))
        else:
            acc = tl.load(local_ptr).to(acc_dtype)
            for reduce_src_rank in tl.static_range(0, world_size):
                if reduce_src_rank != cur_rank:
                    src_offsets = reduce_src_rank * total_elements + linear_offsets
                    src_ptr = remote_inbox + src_offsets
                    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                    acc += tl.load(src_ptr).to(acc_dtype)

        out_ptr = C + linear_offsets
        out_ptr = tl.max_contiguous(tl.multiple_of(out_ptr, BLOCK_SIZE), BLOCK_SIZE)
        tl.store(out_ptr, acc.to(C.type.element_ty))

def _default_chunk_size(total_tiles: int, group_size_m: int, num_xcds: int) -> int:
    chunk_size = group_size_m * group_size_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    return max(1, chunk_size)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return ((max(1, value) + multiple - 1) // multiple) * multiple


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _partitioned_xcd_gemm_num_sms(launch: dict, world_size: int) -> int:
    cached = launch.get("gemm_num_sms")
    if cached is not None:
        return int(cached)
    num_xcds = max(1, launch["num_xcds"])
    return _round_up_to_multiple(max(launch["num_sms"], min(num_xcds, world_size)), num_xcds)


def _default_reduce_num_sms() -> int:
    return 64  # CCL default comm_sms


def _reduce_block_size_n(N: int) -> int:
    # Match CCL default tile size for better performance
    return 64  # CCL default block_size_n


def _make_origami_selector(M: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor, C):

    c_dtype = C.dtype if hasattr(C, "dtype") else C
    return _make_matmul_selector(
        M,
        N,
        K,
        A.dtype,
        B.dtype,
        c_dtype,
        A.device,
        streamk=False,
    )


def _build_partitioned_xcd_transfer_plan(M: int, N: int, world_size: int, launch: dict, element_size: int) -> dict:
    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    group_size_m = launch["group_size_m"]
    num_xcds = max(1, launch["num_xcds"])
    gemm_num_sms = _partitioned_xcd_gemm_num_sms(launch, world_size)
    sms_per_xcd = max(1, gemm_num_sms // num_xcds)
    num_pid_n = _ceil_div(N, block_size_n)
    partition_rows = _ceil_div(M, world_size)

    transfers_by_owner_wave = []
    batch_tile_counts_by_owner = []
    owner_last_wave = []
    max_batches_per_partition = 0

    for owner_rank in range(world_size):
        partition_m_start = owner_rank * partition_rows
        partition_m_end = min(partition_m_start + partition_rows, M)
        first_pid_m = partition_m_start // block_size_m
        last_pid_m = _ceil_div(partition_m_end, block_size_m)
        partition_tiles_m = max(0, last_pid_m - first_pid_m)
        owner_batches = []
        last_wave = -1

        partition_tiles = partition_tiles_m * num_pid_n
        if partition_tiles > 0:
            plan = build_launch_wave_plan(
                num_tiles_m=partition_tiles_m,
                num_tiles_n=num_pid_n,
                group_size_m=group_size_m,
                launch_grid=partition_tiles,
                wave_size=sms_per_xcd,
                num_xcds=1,
                chunk_size=1,
            )
            max_batches_per_partition = max(max_batches_per_partition, plan.num_waves)
            owner_batches = [[] for _ in range(plan.num_waves)]
            owner_counts = list(plan.wave_tile_counts)
            last_wave = max((wave_id for wave_id, count in enumerate(owner_counts) if count), default=-1)

            for transfer in plan.transfers:
                tile_m_start = (first_pid_m + transfer.m_tile_start) * block_size_m
                tile_m_end = min(tile_m_start + transfer.m_tile_count * block_size_m, M)
                seg_start = max(tile_m_start, partition_m_start)
                seg_end = min(tile_m_end, partition_m_end)
                if seg_start >= seg_end:
                    continue

                col = transfer.n_tile_start * block_size_n
                width = min(transfer.n_tile_count * block_size_n, N - col)
                local_m = seg_start - partition_m_start
                owner_batches[transfer.wave_id].append((local_m, col, width * element_size, seg_end - seg_start))
        else:
            owner_counts = []

        transfers_by_owner_wave.append(owner_batches)
        batch_tile_counts_by_owner.append(owner_counts)
        owner_last_wave.append(last_wave)

    max_batches_per_partition = max(1, max_batches_per_partition)

    wave_transfer_offsets = []
    wave_transfer_counts = []
    transfer_row_offsets = []
    transfer_col_offsets = []
    transfer_width_bytes = []
    transfer_heights = []
    flat_batch_tile_counts = []
    max_rects_per_owner_wave = 0
    running_offset = 0

    for owner_rank in range(world_size):
        owner_batches = transfers_by_owner_wave[owner_rank]
        owner_counts = batch_tile_counts_by_owner[owner_rank]
        while len(owner_batches) < max_batches_per_partition:
            owner_batches.append([])
            owner_counts.append(0)

        for batch_transfers, tile_count in zip(owner_batches, owner_counts):
            wave_transfer_offsets.append(running_offset)
            wave_transfer_counts.append(len(batch_transfers))
            flat_batch_tile_counts.append(tile_count)
            max_rects_per_owner_wave = max(max_rects_per_owner_wave, len(batch_transfers))
            for row, col, width_bytes, height in batch_transfers:
                transfer_row_offsets.append(row)
                transfer_col_offsets.append(col)
                transfer_width_bytes.append(width_bytes)
                transfer_heights.append(height)
            running_offset += len(batch_transfers)

    return {
        "gemm_num_sms": gemm_num_sms,
        "sms_per_xcd": sms_per_xcd,
        "max_batches_per_partition": max_batches_per_partition,
        "num_transfer_flags": world_size * max_batches_per_partition,
        "transfers_by_owner_wave": transfers_by_owner_wave,
        "wave_tile_counts": batch_tile_counts_by_owner,
        "flat_wave_tile_counts": flat_batch_tile_counts,
        "wave_transfer_offsets": wave_transfer_offsets,
        "wave_transfer_counts": wave_transfer_counts,
        "owner_last_wave": owner_last_wave,
        "transfer_row_offsets": transfer_row_offsets,
        "transfer_col_offsets": transfer_col_offsets,
        "transfer_width_bytes": transfer_width_bytes,
        "transfer_heights": transfer_heights,
        "max_rects_per_owner_wave": max(1, max_rects_per_owner_wave),
        "num_transfers": running_offset,
    }


def _ensure_transfer_workspace(
    shmem,
    workspace: FusedWorkspace,
    M: int,
    N: int,
    world_size: int,
    launch: dict,
    element_size: int,
    device: torch.device,
):
    gemm_num_sms = _partitioned_xcd_gemm_num_sms(launch, world_size)
    plan_key = (
        "partitioned_xcd",
        M,
        N,
        launch["block_size_m"],
        launch["block_size_n"],
        launch["group_size_m"],
        gemm_num_sms,
        launch["num_xcds"],
        launch["chunk_size"],
        world_size,
        element_size,
    )
    if getattr(workspace, "transfer_plan_key", None) != plan_key:
        plan = _build_partitioned_xcd_transfer_plan(M, N, world_size, launch, element_size)
        workspace.transfer_plan_key = plan_key
        workspace.launch_wave_plan = None
        workspace.num_transfer_waves = plan["max_batches_per_partition"]
        workspace.num_transfer_flags = plan["num_transfer_flags"]
        workspace.batch_tiles_per_xcd = plan["sms_per_xcd"]
        workspace.max_rects_per_owner_wave = plan["max_rects_per_owner_wave"]
        workspace.num_transfers = plan["num_transfers"]
        workspace.transfers_by_owner_wave = plan["transfers_by_owner_wave"]
        workspace.wave_tile_counts_host = plan["wave_tile_counts"]
        workspace.owner_last_wave_host = plan["owner_last_wave"]
        workspace.wave_tile_counts = torch.tensor(plan["flat_wave_tile_counts"], device=device, dtype=torch.int32)
        workspace.wave_transfer_offsets = torch.tensor(plan["wave_transfer_offsets"], device=device, dtype=torch.int32)
        workspace.wave_transfer_counts = torch.tensor(plan["wave_transfer_counts"], device=device, dtype=torch.int32)
        workspace.owner_last_wave = torch.tensor(plan["owner_last_wave"], device=device, dtype=torch.int32)
        workspace.transfer_row_offsets = torch.tensor(plan["transfer_row_offsets"], device=device, dtype=torch.int32)
        workspace.transfer_col_offsets = torch.tensor(plan["transfer_col_offsets"], device=device, dtype=torch.int32)
        workspace.transfer_width_bytes = torch.tensor(plan["transfer_width_bytes"], device=device, dtype=torch.int32)
        workspace.transfer_heights = torch.tensor(plan["transfer_heights"], device=device, dtype=torch.int32)

        workspace.locks = shmem.zeros((plan["num_transfer_flags"],), dtype=torch.int32)

    if getattr(workspace, "completion_signals", None) is None or workspace.completion_signals.numel() != world_size:
        workspace.completion_signals = shmem.zeros((world_size,), dtype=torch.int32)


def _post_host_copy_engine_transfers(
    shmem,
    workspace: FusedWorkspace,
    local_aux_buffer: torch.Tensor,
    reduce_buffer: torch.Tensor,
    rank: int,
    world_size: int,
    rows_per_rank: int,
    flag_iteration: int,
) -> float:
    """Queue host-side SDMA wait+copy packets for the two-shot reduce-scatter."""
    element_size = local_aux_buffer.element_size()
    stride_local_aux_m, stride_local_aux_n = local_aux_buffer.stride()
    stride_reduce_m, stride_reduce_n = reduce_buffer.stride()

    transfers_by_owner_wave = workspace.transfers_by_owner_wave
    wave_tile_counts = workspace.wave_tile_counts_host
    owner_last_wave = workspace.owner_last_wave_host
    max_batches_per_partition = workspace.num_transfer_waves

    local_aux_base = local_aux_buffer.data_ptr()
    reduce_base = reduce_buffer.data_ptr()
    signal_ptr_local = workspace.completion_signals.data_ptr() + rank * workspace.completion_signals.element_size()

    for dst_rank in range(world_size):
        if dst_rank == rank:
            continue

        dst_waves = transfers_by_owner_wave[dst_rank]
        for wave_id, wave_transfers in enumerate(dst_waves):
            if not wave_transfers:
                continue

            wait_value = (flag_iteration + 1) * wave_tile_counts[dst_rank][wave_id]
            flag_id = dst_rank * max_batches_per_partition + wave_id
            wait_flag = workspace.locks.data_ptr() + flag_id * workspace.locks.element_size()
            signal_flag = None
            if wave_id == owner_last_wave[dst_rank]:
                signal_flag = shmem.heap.translate(signal_ptr_local, rank, dst_rank)

            tiles = []
            dst_ptrs = []
            dst_strides = []
            for row_offset, col_offset, width_bytes, height in wave_transfers:
                width_elems = width_bytes // element_size

                tile = sdma_ep.Tile()
                tile.pid_m = 0
                tile.pid_n = 0
                tile.block_m = height
                tile.block_n = width_elems
                tile.elem_size = element_size
                tile.src_stride = stride_local_aux_m * element_size

                src_offset = (
                    (dst_rank * rows_per_rank + row_offset) * stride_local_aux_m
                    + col_offset * stride_local_aux_n
                )
                tile.data = local_aux_base + src_offset * element_size

                dst_offset = (rank * rows_per_rank + row_offset) * stride_reduce_m + col_offset * stride_reduce_n
                dst_ptr_local = reduce_base + dst_offset * element_size

                tiles.append(tile)
                dst_ptrs.append(shmem.heap.translate(dst_ptr_local, rank, dst_rank))
                dst_strides.append(stride_reduce_m * element_size)

            shmem.put_tiles(
                tiles,
                dst_rank=dst_rank,
                dst_ptrs=dst_ptrs,
                dst_strides=dst_strides,
                wait_flag=wait_flag,
                wait_value=wait_value,
                signal_flag=signal_flag,
                signal_value=flag_iteration + 1,
                async_op=True,
                channel=0,
            )



def _post_host_copy_engine_broadcast_transfers(
    shmem,
    workspace: FusedWorkspace,
    local_aux_buffer: torch.Tensor,
    remote_inbox: torch.Tensor,
    rank: int,
    world_size: int,
    M: int,
    flag_iteration: int,
) -> float:
    """Queue host-side SDMA wait+copy packets for the one-shot all-gather of GEMM partials."""
    element_size = local_aux_buffer.element_size()
    stride_local_aux_m, stride_local_aux_n = local_aux_buffer.stride()
    stride_remote_m, stride_remote_n = remote_inbox.stride()

    transfers_by_partition_wave = workspace.transfers_by_owner_wave
    wave_tile_counts = workspace.wave_tile_counts_host
    max_batches_per_partition = workspace.num_transfer_waves
    partition_rows = _ceil_div(M, world_size)

    local_aux_base = local_aux_buffer.data_ptr()
    remote_base = remote_inbox.data_ptr()
    signal_ptr_local = workspace.completion_signals.data_ptr() + rank * workspace.completion_signals.element_size()

    transfer_work = []
    for partition_id, partition_waves in enumerate(transfers_by_partition_wave):
        partition_m_start = partition_id * partition_rows
        for wave_id, wave_transfers in enumerate(partition_waves):
            if wave_transfers:
                transfer_work.append((partition_id, partition_m_start, wave_id, wave_transfers))

    for dst_rank in range(world_size):
        if dst_rank == rank:
            continue

        for work_idx, (partition_id, partition_m_start, wave_id, wave_transfers) in enumerate(transfer_work):
            wait_value = (flag_iteration + 1) * wave_tile_counts[partition_id][wave_id]
            flag_id = partition_id * max_batches_per_partition + wave_id
            wait_flag = workspace.locks.data_ptr() + flag_id * workspace.locks.element_size()
            signal_flag = None
            if work_idx == len(transfer_work) - 1:
                signal_flag = shmem.heap.translate(signal_ptr_local, rank, dst_rank)

            tiles = []
            dst_ptrs = []
            dst_strides = []
            for row_offset, col_offset, width_bytes, height in wave_transfers:
                width_elems = width_bytes // element_size
                global_row = partition_m_start + row_offset

                tile = sdma_ep.Tile()
                tile.pid_m = 0
                tile.pid_n = 0
                tile.block_m = height
                tile.block_n = width_elems
                tile.elem_size = element_size
                tile.src_stride = stride_local_aux_m * element_size

                src_offset = global_row * stride_local_aux_m + col_offset * stride_local_aux_n
                tile.data = local_aux_base + src_offset * element_size

                dst_offset = (rank * M + global_row) * stride_remote_m + col_offset * stride_remote_n
                dst_ptr_local = remote_base + dst_offset * element_size

                tiles.append(tile)
                dst_ptrs.append(shmem.heap.translate(dst_ptr_local, rank, dst_rank))
                dst_strides.append(stride_remote_m * element_size)

            shmem.put_tiles(
                tiles,
                dst_rank=dst_rank,
                dst_ptrs=dst_ptrs,
                dst_strides=dst_strides,
                wait_flag=wait_flag,
                wait_value=wait_value,
                signal_flag=signal_flag,
                signal_value=flag_iteration + 1,
                async_op=True,
                channel=0,
            )



def matmul_all_reduce_copy_engine_prepost_transfers(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: FusedWorkspace,
    flag_iteration: int = 0,
) -> float:
    """Queue SDMA wait+copy packets before the timed GEMM launch."""
    if workspace is None:
        raise ValueError("workspace is required when preposting copy-engine transfers")
    if workspace.variant not in ("one_shot", "two_shot"):
        return 0.0
    if workspace.a_inbox is None:
        raise ValueError("copy-engine workspace must have a_inbox before preposting")
    if workspace.variant == "two_shot" and workspace.aux_buffer is None:
        raise ValueError("two_shot copy-engine workspace must have aux_buffer before preposting")

    M, _ = A.shape
    N = B.shape[1]
    world_size = shmem.get_num_ranks()
    launch = workspace.launch_params
    if launch is None:
        raise ValueError("workspace.launch_params must be initialized before preposting")

    _ensure_transfer_workspace(shmem, workspace, M, N, world_size, launch, A.element_size(), A.device)
    if workspace.variant == "one_shot":
        rank = shmem.get_rank()
        local_inbox_slice = workspace.a_inbox[rank * M : (rank + 1) * M, :]
        return _post_host_copy_engine_broadcast_transfers(
            shmem,
            workspace,
            local_inbox_slice,
            workspace.a_inbox,
            rank,
            world_size,
            M,
            flag_iteration,
        )

    return _post_host_copy_engine_transfers(
        shmem,
        workspace,
        workspace.aux_buffer,
        workspace.a_inbox,
        shmem.get_rank(),
        world_size,
        M // world_size,
        flag_iteration,
    )


def _selector_active_cus(selector, device: torch.device) -> int:
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        props = torch.cuda.get_device_properties(device)
        active_cus = props.multi_processor_count
    return int(active_cus)


def _matmul_all_reduce_copy_engine_launch_params(
    M: int,
    N: int,
    selector,
    device: torch.device,
) -> dict:
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    num_stages = getattr(selector, "num_stages", 2)

    # Origami calls this num_sms, but it is the XCD/chiplet workgroup mapping
    # count used by chiplet_transform_chunked, not the persistent launch grid.
    num_xcds = selector.num_sms
    if num_xcds <= 0:
        num_xcds = 1

    num_tiles_m = (M + block_size_m - 1) // block_size_m
    num_tiles_n = (N + block_size_n - 1) // block_size_n
    total_tiles = num_tiles_m * num_tiles_n
    num_sms = min(_selector_active_cus(selector, device), total_tiles)
    chunk_size = _default_chunk_size(num_sms, group_size_m, num_xcds)

    return {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_xcds": num_xcds,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "total_tiles": total_tiles,
        "num_sms": num_sms,
        "chunk_size": chunk_size,
        "num_warps": 8,
        "num_stages": num_stages,
        "matrix_instr_nonkdim": 16,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "reduce_block_size_m": 32,  # Match CCL default block_size_m
        "reduce_block_size_n": _reduce_block_size_n(N),
        "reduce_num_sms": _default_reduce_num_sms(),
    }


def matmul_all_reduce_copy_engine_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    selector=None,
    out_dtype: Optional[torch.dtype] = None,
) -> FusedWorkspace:
    """
    Allocate and reset temporary buffers for matmul_all_reduce_copy_engine.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N)
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        config: Optional FusedConfig. If None, uses defaults.
        workspace: Optional existing workspace to reuse. If None, creates new one.
        selector: Optional pre-built tritonBLAS Origami selector.
        out_dtype: Optional output dtype for selector construction.

    Returns:
        FusedWorkspace instance ready for kernel launch.
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape[:2]
    N = B.shape[1]
    dtype = A.dtype
    world_size = shmem.get_num_ranks()

    # Validate config
    config.validate(world_size=world_size)

    if config.all_reduce_variant == "two_shot" and M % world_size != 0:
        raise ValueError(
            "matmul_all_reduce_copy_engine two_shot requires M to be divisible by world_size "
            "because the final all-gather uses equal row shards."
        )

    if selector is None:
        c_dtype = dtype if out_dtype is None else out_dtype
        selector = _make_origami_selector(M, N, K, A, B, c_dtype)

    launch = _matmul_all_reduce_copy_engine_launch_params(M, N, selector, A.device)

    launch["gemm_num_sms"] = _partitioned_xcd_gemm_num_sms(launch, world_size)
    if config.all_reduce_variant == "two_shot":
        launch["reduce_num_sms"] = _default_reduce_num_sms()
    else:
        launch["reduce_num_sms"] = launch["num_sms"]

    if workspace is None:
        workspace = FusedWorkspace()

    workspace.operation = "matmul_all_reduce_copy_engine"
    workspace.shape = (M, N, K)
    workspace.dtype = dtype
    workspace.world_size = world_size
    workspace.variant = config.all_reduce_variant
    workspace.selector = selector
    workspace.config = config
    workspace.launch_params = launch

    aux_rows = M

    if config.all_reduce_variant == "two_shot":
        if workspace.aux_buffer is None or workspace.aux_buffer.shape != (aux_rows, N):
            workspace.aux_buffer = shmem.zeros((aux_rows, N), dtype=dtype)
        else:
            workspace.aux_buffer.zero_()
        inbox_rows = aux_rows
    else:
        workspace.aux_buffer = None
        inbox_rows = aux_rows * world_size

    if workspace.a_inbox is None or workspace.a_inbox.shape != (inbox_rows, N):
        workspace.a_inbox = shmem.zeros((inbox_rows, N), dtype=dtype)
    else:
        workspace.a_inbox.zero_()
    _ensure_transfer_workspace(shmem, workspace, M, N, world_size, launch, A.element_size(), A.device)
    workspace.locks.zero_()
    workspace.completion_signals.zero_()

    # Zero output tensor
    C.zero_()

    return workspace


def matmul_all_reduce_copy_engine(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    selector=None,
    flag_iteration: int = 0,
    copy_engine_transfers_preposted: bool = False,
    split_completion_wait: bool = False,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-reduce using SDMA copy-engine transfers.

    Computes: C = all_reduce(A @ B) across all ranks. The one_shot variant
    broadcasts every rank's GEMM output to all peers and reduces locally; the
    two_shot variant copies row shards to owner ranks, reduces them, then
    all-gathers the reduced shards.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M, N) - will contain reduced result on all ranks
        A: Input matrix A (M, K) - each rank has different data (data-parallel)
        B: Input matrix B (K, N) - replicated across ranks
        async_op: If False, performs barrier at end. Default: False.
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional pre-allocated workspace. If None, creates new one.
        selector: Optional pre-built tritonBLAS Origami selector.
        flag_iteration: Launch generation for cumulative copy-engine wait and
            completion counters. Increment this when reusing a workspace without
            zeroing its synchronization buffers.
        copy_engine_transfers_preposted: If True, the SDMA wait+copy
            packets for this flag_iteration were already queued by the caller,
            usually from a benchmark preamble_fn outside the timed region.
        split_completion_wait: If True, launch a separate completion-wait
            kernel and run the reduce kernel with its inline completion wait
            disabled. This is intended for profiling wait time, not the hot
            benchmark path.

    Returns:
        workspace: Updated workspace object (can be reused for subsequent calls)

    Example:
        >>> A = shmem.randn((1024, 512), dtype=torch.float16)
        >>> B = shmem.randn((512, 2048), dtype=torch.float16)
        >>> C = shmem.zeros((1024, 2048), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce_copy_engine(C, A, B)
    """
    if config is None:
        config = FusedConfig()

    # Extract dimensions
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got shapes {A.shape} and {B.shape}")

    M, K = A.shape
    K_B, N = B.shape

    if K != K_B:
        raise ValueError(
            f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K_B}, {N}). "
            f"Inner dimensions must match (K={K} != K_B={K_B})"
        )

    if C.shape != (M, N):
        raise ValueError(f"Output tensor shape {C.shape} doesn't match expected ({M}, {N})")

    if A.dtype != B.dtype or A.dtype != C.dtype:
        raise ValueError(f"All tensors must have same dtype, got A:{A.dtype}, B:{B.dtype}, C:{C.dtype}")

    # Extract strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Get rank info
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    config.validate(world_size=world_size)

    # Prepare workspace if needed
    if workspace is None:
        workspace = matmul_all_reduce_copy_engine_preamble(
            shmem,
            C,
            A,
            B,
            config=config,
            workspace=workspace,
            selector=selector,
            out_dtype=C.dtype,
        )

    launch = workspace.launch_params

    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    block_size_k = launch["block_size_k"]

    _ensure_transfer_workspace(shmem, workspace, M, N, world_size, launch, A.element_size(), A.device)

    even_k = K % block_size_k == 0
    launch_kwargs = {
        "num_warps": launch["num_warps"],
        "matrix_instr_nonkdim": launch["matrix_instr_nonkdim"],
    }
    if launch["num_stages"] is not None:
        launch_kwargs["num_stages"] = launch["num_stages"]

    if config.all_reduce_variant == "one_shot":
        # Rank-major all-inbox layout:
        # a_inbox[src_rank * M + row, col] holds src_rank's GEMM partial.
        gemm_output_buffer = workspace.a_inbox[rank * M : (rank + 1) * M, :]
    else:
        gemm_output_buffer = workspace.aux_buffer
    stride_local_aux_m, stride_local_aux_n = gemm_output_buffer.stride()
    reduce_buffer = workspace.a_inbox
    stride_reduce_m, stride_reduce_n = reduce_buffer.stride()

    gemm_num_sms = launch.get("gemm_num_sms")
    if gemm_num_sms is None:
        gemm_num_sms = _partitioned_xcd_gemm_num_sms(launch, world_size)
        launch["gemm_num_sms"] = gemm_num_sms
    iris_launch(
        _partitioned_xcd_matmul_kernel,
        (gemm_num_sms,),
        A,
        B,
        gemm_output_buffer,
        reduce_buffer,
        workspace.locks,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_local_aux_m,
        stride_local_aux_n,
        stride_reduce_m,
        stride_reduce_n,
        rank,
        block_size_m,
        block_size_n,
        block_size_k,
        launch["group_size_m"],
        gemm_num_sms,
        launch["num_xcds"],
        launch["chunk_size"],
        world_size,
        workspace.num_transfer_waves,
        config.all_reduce_variant == "two_shot",
        even_k,
        launch["allow_tf32"],
        algorithm="matmul_all_reduce_copy_engine_partitioned_xcd_gemm",
        rank=rank,
        dtype=A.dtype,
        **launch_kwargs,
    )

    if not copy_engine_transfers_preposted:
        matmul_all_reduce_copy_engine_prepost_transfers(
            shmem,
            A,
            B,
            workspace,
            flag_iteration,
        )

    if config.all_reduce_variant == "one_shot":
        reduce_block_size_m = 1
        reduce_block_size_n = 512
        num_sms = max(launch["reduce_num_sms"], launch["num_sms"] * 3)
        wait_for_completion = True
        if split_completion_wait:
            iris_launch(
                _matmul_all_reduce_copy_engine_wait_completion_kernel,
                (world_size,),
                workspace.completion_signals,
                flag_iteration + 1,
                cur_rank=rank,
                world_size=world_size,
                algorithm="matmul_all_reduce_copy_engine_wait_completion",
                rank=rank,
                dtype=A.dtype,
                num_warps=1,
            )
            wait_for_completion = False

        total_reduce_elements = M * N
        use_flat_local_reduce = (
            C.is_contiguous()
            and gemm_output_buffer.is_contiguous()
            and workspace.a_inbox.is_contiguous()
            and total_reduce_elements % reduce_block_size_n == 0
        )

        if use_flat_local_reduce:
            total_reduce_blocks = total_reduce_elements // reduce_block_size_n
            reduce_grid = (min(num_sms, total_reduce_blocks),)
            iris_launch(
                _matmul_all_reduce_copy_engine_local_reduce_flat_kernel,
                reduce_grid,
                C,
                gemm_output_buffer,
                workspace.a_inbox,
                workspace.completion_signals,
                flag_iteration + 1,
                total_reduce_elements,
                cur_rank=rank,
                world_size=world_size,
                BLOCK_SIZE=reduce_block_size_n,
                NUM_SMS=num_sms,
                WAIT_FOR_COMPLETION=wait_for_completion,
                algorithm="matmul_all_reduce_copy_engine_local_reduce_flat",
                rank=rank,
                dtype=A.dtype,
                num_warps=4,
                num_stages=2,
            )
        else:
            reduce_tiles_m = (M + reduce_block_size_m - 1) // reduce_block_size_m
            reduce_tiles_n = (N + reduce_block_size_n - 1) // reduce_block_size_n
            total_reduce_tiles = reduce_tiles_m * reduce_tiles_n
            reduce_grid = (min(num_sms, total_reduce_tiles),)
            iris_launch(
                _matmul_all_reduce_copy_engine_local_reduce_kernel,
                reduce_grid,
                C,
                gemm_output_buffer,
                workspace.a_inbox,
                workspace.completion_signals,
                flag_iteration + 1,
                M,
                N,
                gemm_output_buffer.stride(0),
                gemm_output_buffer.stride(1),
                workspace.a_inbox.stride(0),
                workspace.a_inbox.stride(1),
                C.stride(0),
                C.stride(1),
                cur_rank=rank,
                world_size=world_size,
                REDUCE_BLOCK_SIZE_M=reduce_block_size_m,
                REDUCE_BLOCK_SIZE_N=reduce_block_size_n,
                GROUP_SIZE_M=launch["group_size_m"],
                NUM_SMS=num_sms,
                WAIT_FOR_COMPLETION=wait_for_completion,
                algorithm="matmul_all_reduce_copy_engine_local_reduce",
                rank=rank,
                dtype=A.dtype,
                num_warps=4,
                num_stages=1,
            )

    if config.all_reduce_variant == "two_shot":
        reduce_block_size_m = launch["reduce_block_size_m"]
        reduce_block_size_n = launch["reduce_block_size_n"]
        rows_per_rank = M // world_size
        reduce_tiles_m = (rows_per_rank + reduce_block_size_m - 1) // reduce_block_size_m
        reduce_tiles_n = (N + reduce_block_size_n - 1) // reduce_block_size_n
        total_reduce_tiles = reduce_tiles_m * reduce_tiles_n

        # Use persistent kernel with limited number of workgroups
        num_sms = launch["reduce_num_sms"]
        reduce_grid = (min(num_sms, total_reduce_tiles),)

        # Reduce-scatter waits for the same completion condition as
        # _matmul_all_reduce_copy_engine_wait_completion_kernel: every remote
        # rank's completion slot must reach flag_iteration + 1.
        device_context = shmem.get_device_context()
        iris_launch(
            _matmul_all_reduce_copy_engine_reduce_scatter_kernel,
            reduce_grid,
            C,
            workspace.a_inbox,
            workspace.completion_signals,
            flag_iteration + 1,
            M,
            N,
            workspace.a_inbox.stride(0),
            workspace.a_inbox.stride(1),
            C.stride(0),
            C.stride(1),
            device_context,
            cur_rank=rank,
            world_size=world_size,
            REDUCE_BLOCK_SIZE_M=launch["reduce_block_size_m"],
            REDUCE_BLOCK_SIZE_N=launch["reduce_block_size_n"],
            GROUP_SIZE_M=launch["group_size_m"],
            NUM_SMS=num_sms,
            WAIT_FOR_COMPLETION=True,
            algorithm="matmul_all_reduce_copy_engine_reduce_scatter",
            rank=rank,
            dtype=A.dtype,
            num_warps=4,
        )

    # Barrier unless async
    if not async_op:
        shmem.barrier()

    return workspace
