# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused matrix multiplication and reduce-scatter.

Rank ``r`` owns rows ``[r * M / W, (r + 1) * M / W)`` of ``A @ B`` (``W`` ranks) and gets their
sum over all ranks. Two stages, which the two-shot ``matmul_all_reduce`` can share:

1. ``_publish_kernel``: the GEMM stores each owner's partial rows into this rank's slot of the
   owner's inbox (``inbox[src * M / W + row]``). Owners take turns one strip (``BLOCK_SIZE_M`` rows,
   all of N) at a time, this rank's own rows last. Each rank signals each owner once its partials
   have all arrived.
2. ``wait_inbox`` + ``sum_inbox_tile``: wait for every rank's partials, then sum the slots in rank
   order, in fp32.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, make_tensor_view, Tile

from .config import FusedConfig
from .workspace import FusedWorkspace
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.mem.triton.context import __translate
from iris.mem.utils import wait_cnt

REDUCE_BLOCK_SIZE_M = 32
REDUCE_BLOCK_SIZE_N = 256


@triton.jit
def _publish_kernel(
    A,
    B,
    inbox,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    max_strips,
    heap_bases,
    arrivals,
    partition_counters,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Program ``i`` computes the ``i``-th tile of this rank's order: owners' strips round-robin
    (starting at the next rank), tiles of a strip left to right, this rank's own strips last. A tile
    row that straddles two owners is computed once for each, masked to that owner's rows.
    """
    step = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    rows_per_rank = M // world_size
    if world_size == 1:
        owner = cur_rank
        strip = step // num_pid_n
    else:
        others = (world_size - 1) * max_strips * num_pid_n
        c = step // num_pid_n
        j = tl.where(step < others, c % (world_size - 1), world_size - 1)
        strip = tl.where(step < others, c // (world_size - 1), (step - others) // num_pid_n)
        owner = (cur_rank + 1 + j) % world_size
    part_start = owner * rows_per_rank
    part_end = part_start + rows_per_rank
    first_pid_m = part_start // BLOCK_SIZE_M
    num_strips = tl.cdiv(part_end, BLOCK_SIZE_M) - first_pid_m

    # Owners whose rows span fewer tile rows than max_strips leave some programs idle
    if strip < num_strips:
        pid_m = first_pid_m + strip
        pid_n = step % num_pid_n
        tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
        tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
        # No cache modifiers: ".cg" loads (GemmContext's default) make this GEMM ~10% slower
        gemm_ctx = GemmContext(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_sms=1,
            cache_modifier_a=None,
            cache_modifier_b=None,
            even_k=EVEN_K,
        )
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N))
        c_tile = acc.to(inbox.type.element_ty)

        rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = ((rows >= part_start) & (rows < part_end))[:, None] & (cols[None, :] < N)
        # This rank's slot in the owner's inbox
        slot_rows = cur_rank * rows_per_rank + rows - part_start
        slot_offsets = slot_rows[:, None] * N + cols[None, :]

        if owner == cur_rank:
            tl.store(inbox + slot_offsets, c_tile, mask=mask)
        else:
            dst = __translate(inbox, cur_rank, owner, heap_bases)
            tl.store(dst + slot_offsets, c_tile, mask=mask)
            # Every wave's remote stores are done before the ticket; the owner's last tile signals
            wait_cnt()
            tl.debug_barrier()
            tiles = num_strips * num_pid_n
            ticket = tl.atomic_add(partition_counters + owner, 1, sem="acq_rel", scope="sys")
            if ticket % tiles == tiles - 1:
                iris.atomic_add(arrivals + cur_rank, 1, cur_rank, owner, heap_bases, sem="release", scope="sys")


@triton.jit
def wait_inbox(arrivals, target, cur_rank, world_size: tl.constexpr):
    """Spin until every other rank's partials for this call have arrived in the inbox."""
    for s in tl.static_range(world_size):
        if s != cur_rank:
            while tl.atomic_add(arrivals + s, 0, sem="acquire", scope="sys") < target:
                pass


@triton.jit
def sum_inbox_tile(inbox, rows_per_rank, N, rows, cols, mask, world_size: tl.constexpr):
    """Sum of every rank's slot at local ``rows`` x ``cols`` of the inbox, in rank order, in fp32."""
    total = tl.zeros((rows.shape[0], cols.shape[0]), dtype=tl.float32)
    for s in tl.static_range(world_size):
        total += tl.load(inbox + (s * rows_per_rank + rows)[:, None] * N + cols[None, :], mask=mask).to(tl.float32)
    return total


@triton.jit
def _reduce_kernel(
    inbox,
    C,
    rows_per_rank,
    N,
    stride_cm,
    stride_cn,
    arrivals,
    target,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
):
    """Write this rank's rows: the sum of every rank's slot."""
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, REDUCE_BLOCK_SIZE_N)
    rows = (pid // tiles_n) * REDUCE_BLOCK_SIZE_M + tl.arange(0, REDUCE_BLOCK_SIZE_M)
    cols = (pid % tiles_n) * REDUCE_BLOCK_SIZE_N + tl.arange(0, REDUCE_BLOCK_SIZE_N)
    mask = (rows[:, None] < rows_per_rank) & (cols[None, :] < N)
    wait_inbox(arrivals, target, cur_rank, world_size)
    total = sum_inbox_tile(inbox, rows_per_rank, N, rows, cols, mask, world_size)
    c_ptr = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptr, total.to(C.type.element_ty), mask=mask)


def _max_strips(rows_per_rank, block_size_m):
    """Tile rows an owner's rows can span: one more when they don't start on a tile boundary."""
    if rows_per_rank % block_size_m == 0:
        return rows_per_rank // block_size_m
    return triton.cdiv(rows_per_rank, block_size_m) + 1


def matmul_reduce_scatter_preamble(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Allocate the inbox and counters, unless ``workspace`` already fits. Collective when it allocates.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M / world_size, N)
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        config: Optional FusedConfig. If None, uses defaults.
        workspace: Optional existing workspace to reuse. If None, creates new one.

    Returns:
        FusedWorkspace instance ready for kernel launch.
    """
    if config is None:
        config = FusedConfig()
    world_size = shmem.get_num_ranks()
    config.validate(world_size=world_size)
    if workspace is None:
        workspace = FusedWorkspace()

    M, K = A.shape[:2]
    N = B.shape[1]
    if M % world_size:
        raise ValueError(f"M ({M}) must be a multiple of the number of ranks ({world_size})")
    if tuple(C.shape) != (M // world_size, N):
        raise ValueError(f"C has shape {tuple(C.shape)}, expected (M / world_size, N) = ({M // world_size}, {N})")
    if C.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"matmul_reduce_scatter supports float16, bfloat16 and float32 outputs, got {C.dtype}")

    key = ((M, N, K), config.block_size_m, C.dtype, world_size)
    if workspace.operation == "matmul_reduce_scatter" and workspace.rs_key == key:
        return workspace
    rows_per_rank = M // world_size
    workspace.operation = "matmul_reduce_scatter"
    workspace.shape = (M, N, K)
    workspace.dtype = C.dtype
    workspace.world_size = world_size
    workspace.variant = "two_shot"
    workspace.inbox = shmem.zeros((world_size * rows_per_rank, N), dtype=C.dtype)
    workspace.arrivals = shmem.zeros((world_size,), dtype=torch.int64)
    workspace.partition_counters = torch.zeros(world_size, dtype=torch.int64, device=A.device)
    workspace.round = 0
    workspace.rs_key = key
    workspace.prepared = True
    # Peers may send as soon as they start the first call: every rank's counters must be zero first
    shmem.barrier()
    return workspace


def matmul_reduce_scatter(
    shmem,
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and reduce-scatter.

    Computes ``A @ B`` on every rank and gives rank ``r`` rows ``[r * M / W, (r + 1) * M / W)`` of
    their sum over the ``W`` ranks, like ``torch.distributed.reduce_scatter_tensor`` of ``A @ B``.
    ``M`` must be a multiple of ``W``.

    Pass the returned workspace back in on later calls: a new one allocates buffers and synchronizes
    the ranks. With ``async_op=True``, synchronize the ranks before reusing the workspace.

    Args:
        shmem: Iris shmem context
        C: Output tensor (M / world_size, N)
        A: Input matrix A (M, K)
        B: Input matrix B (K, N)
        async_op: If True, returns immediately without synchronization
        config: Optional FusedConfig for tuning. If None, uses defaults.
        workspace: Optional workspace to reuse. If None, allocates new.

    Returns:
        FusedWorkspace with allocated temporary buffers.
    """
    if config is None:
        config = FusedConfig()
    workspace = matmul_reduce_scatter_preamble(shmem, C, A, B, config, workspace)

    M, K = A.shape[:2]
    N = B.shape[1]
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    rows_per_rank = M // world_size
    max_strips = _max_strips(rows_per_rank, config.block_size_m)
    workspace.round += 1

    from iris.host.logging.logging import _log_rank

    _log_rank(
        logging.DEBUG,
        "matmul_reduce_scatter: shape=(%d,%d,%d) dtype=%s rank=%d/%d",
        M,
        N,
        K,
        A.dtype,
        rank,
        world_size,
        rank=rank,
        num_ranks=world_size,
    )

    heap_bases = shmem.get_heap_bases()
    iris_launch(
        _publish_kernel,
        (world_size * max_strips * triton.cdiv(N, config.block_size_n),),
        A,
        B,
        workspace.inbox,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        max_strips,
        heap_bases,
        workspace.arrivals,
        workspace.partition_counters,
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        K % config.block_size_k == 0,
        num_warps=8,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=1,
        algorithm="matmul_reduce_scatter",
        rank=rank,
        dtype=A.dtype,
    )
    iris_launch(
        _reduce_kernel,
        (triton.cdiv(rows_per_rank, REDUCE_BLOCK_SIZE_M) * triton.cdiv(N, REDUCE_BLOCK_SIZE_N),),
        workspace.inbox,
        C,
        rows_per_rank,
        N,
        C.stride(0),
        C.stride(1),
        workspace.arrivals,
        workspace.round,
        rank,
        world_size,
        REDUCE_BLOCK_SIZE_M,
        REDUCE_BLOCK_SIZE_N,
        num_warps=4,
        algorithm="matmul_reduce_scatter",
        rank=rank,
        dtype=A.dtype,
    )

    if not async_op:
        torch.cuda.synchronize()
        shmem.barrier()

    return workspace
