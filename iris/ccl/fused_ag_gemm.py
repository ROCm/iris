# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM (prologue fusion) for iris-ccl.

Computes output = all_gather(local_shard, dim=1) @ weight by pulling
each rank's K_local shard via XGMI and accumulating partial matmuls
on-the-fly. Uses raw Triton + iris.load — no tritonblas dependency.

Pull pattern: each SM reads remote shards directly from the source
rank's symmetric heap over XGMI, avoiding explicit staging buffers.
Local rank is processed first to minimize XGMI traffic.
"""

import triton
import triton.language as tl
import iris
from .config import Config
from .utils import extract_group_info, chiplet_transform_chunked


@triton.jit()
def _fused_ag_gemm_kernel(
    shard_ptr,
    weight_ptr,
    output_ptr,
    M,
    N,
    K,
    K_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Fused all-gather + GEMM persistent kernel.

    Tiles the output C[M, N] with persistent scheduling. For each output
    tile, loops over all world_size ranks, reading each rank's K_local
    shard of A via iris.load (XGMI) and accumulating A_shard @ W_slice.

    Local rank is processed first (direct HBM, no XGMI translation).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    # Group-relative index of the local rank
    group_rank = (iris_rank - rank_start) // rank_stride

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Swizzled tile indexing for L2 locality
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Initialize fp32 accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Row indices for this output tile
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

        # Column indices for weight / output
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        is_full_m = (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M) <= M
        is_full_n = (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N) <= N
        is_full = is_full_m & is_full_n

        # ── Fast path: no boundary masks needed ──────────────────────────
        if is_full:
            # Local rank first (direct HBM, no XGMI) — rank index = group_rank
            acc = _accumulate_rank_fast(
                shard_ptr, weight_ptr, acc,
                rm, rn, K_local,
                stride_am, stride_ak, stride_bk, stride_bn,
                iris_rank, iris_rank, group_rank, heap_bases,
                BLOCK_SIZE_K, NUM_K_BLOCKS_LOCAL, EVEN_K, K,
                is_local=True,
            )

            # Remote ranks via XGMI
            for j in tl.static_range(1, world_size):
                source_rank_idx = (group_rank + j) % world_size
                source_rank_global = rank_start + source_rank_idx * rank_stride
                acc = _accumulate_rank_fast(
                    shard_ptr, weight_ptr, acc,
                    rm, rn, K_local,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    iris_rank, source_rank_global, source_rank_idx, heap_bases,
                    BLOCK_SIZE_K, NUM_K_BLOCKS_LOCAL, EVEN_K, K,
                    is_local=False,
                )

        # ── Slow path: masked boundary tiles ─────────────────────────────
        else:
            m_mask = rm < M
            n_mask = rn < N

            # Local rank first
            acc = _accumulate_rank_slow(
                shard_ptr, weight_ptr, acc,
                rm, rn, m_mask, n_mask, K_local,
                stride_am, stride_ak, stride_bk, stride_bn,
                iris_rank, iris_rank, group_rank, heap_bases,
                BLOCK_SIZE_K, NUM_K_BLOCKS_LOCAL, EVEN_K, K,
                is_local=True,
            )

            # Remote ranks
            for j in tl.static_range(1, world_size):
                source_rank_idx = (group_rank + j) % world_size
                source_rank_global = rank_start + source_rank_idx * rank_stride
                acc = _accumulate_rank_slow(
                    shard_ptr, weight_ptr, acc,
                    rm, rn, m_mask, n_mask, K_local,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    iris_rank, source_rank_global, source_rank_idx, heap_bases,
                    BLOCK_SIZE_K, NUM_K_BLOCKS_LOCAL, EVEN_K, K,
                    is_local=False,
                )

        # Store output
        c = acc.to(output_ptr.type.element_ty)
        c_ptrs = output_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        if is_full:
            tl.store(c_ptrs, c)
        else:
            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)


@triton.jit()
def _accumulate_rank_fast(
    shard_ptr, weight_ptr, acc,
    rm, rn, K_local,
    stride_am, stride_ak, stride_bk, stride_bn,
    iris_rank, source_rank_global, source_rank_idx, heap_bases,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    EVEN_K: tl.constexpr,
    K,
    is_local: tl.constexpr,
):
    """Accumulate one rank's contribution — unmasked fast path."""
    loop_k = NUM_K_BLOCKS_LOCAL if EVEN_K else NUM_K_BLOCKS_LOCAL - 1

    for k_block in range(0, loop_k):
        k_off = k_block * BLOCK_SIZE_K
        rk = k_off + tl.arange(0, BLOCK_SIZE_K)
        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

        # Load A shard tile
        a_ptrs = shard_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        if is_local:
            a = tl.load(a_ptrs)
        else:
            a = iris.load(a_ptrs, iris_rank, source_rank_global, heap_bases, hint=(1, BLOCK_SIZE_K))

        # Load weight tile (always local — weight is replicated)
        # source_rank_idx is the group-relative index (0..world_size-1)
        global_k = source_rank_idx * K_local + k_off
        rk_global = global_k + tl.arange(0, BLOCK_SIZE_K)
        rk_global = tl.max_contiguous(tl.multiple_of(rk_global, BLOCK_SIZE_K), BLOCK_SIZE_K)
        b_ptrs = weight_ptr + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_ptrs)

        acc = tl.dot(a, b, acc, allow_tf32=True)

    # Remainder K block
    if not EVEN_K:
        k_off = loop_k * BLOCK_SIZE_K
        rk = k_off + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = shard_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = rk[None, :] < K_local
        if is_local:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        else:
            a = iris.load(a_ptrs, iris_rank, source_rank_global, heap_bases, mask=a_mask, hint=(1, BLOCK_SIZE_K))

        global_k = source_rank_idx * K_local + k_off
        rk_global = global_k + tl.arange(0, BLOCK_SIZE_K)
        b_ptrs = weight_ptr + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = rk_global[:, None] < K
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=True)

    return acc


@triton.jit()
def _accumulate_rank_slow(
    shard_ptr, weight_ptr, acc,
    rm, rn, m_mask, n_mask, K_local,
    stride_am, stride_ak, stride_bk, stride_bn,
    iris_rank, source_rank_global, source_rank_idx, heap_bases,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    EVEN_K: tl.constexpr,
    K,
    is_local: tl.constexpr,
):
    """Accumulate one rank's contribution — masked slow path for boundary tiles."""
    loop_k = NUM_K_BLOCKS_LOCAL if EVEN_K else NUM_K_BLOCKS_LOCAL - 1

    for k_block in range(0, loop_k):
        k_off = k_block * BLOCK_SIZE_K
        rk = k_off + tl.arange(0, BLOCK_SIZE_K)

        # Load A shard tile with M mask
        a_ptrs = shard_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask_full = m_mask[:, None]
        if is_local:
            a = tl.load(a_ptrs, mask=a_mask_full, other=0.0)
        else:
            a = iris.load(a_ptrs, iris_rank, source_rank_global, heap_bases, mask=a_mask_full, hint=(1, BLOCK_SIZE_K))

        # Load weight tile with N mask
        global_k = source_rank_idx * K_local + k_off
        rk_global = global_k + tl.arange(0, BLOCK_SIZE_K)
        b_ptrs = weight_ptr + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask_full = n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask_full, other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=True)

    # Remainder K block
    if not EVEN_K:
        k_off = loop_k * BLOCK_SIZE_K
        rk = k_off + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = shard_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask_full = m_mask[:, None] & (rk[None, :] < K_local)
        if is_local:
            a = tl.load(a_ptrs, mask=a_mask_full, other=0.0)
        else:
            a = iris.load(a_ptrs, iris_rank, source_rank_global, heap_bases, mask=a_mask_full, hint=(1, BLOCK_SIZE_K))

        global_k = source_rank_idx * K_local + k_off
        rk_global = global_k + tl.arange(0, BLOCK_SIZE_K)
        b_ptrs = weight_ptr + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask_full = (rk_global[:, None] < K) & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask_full, other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=True)

    return acc


def all_gather_gemm(
    output_tensor,
    local_shard,
    weight,
    shmem,
    group=None,
    async_op=False,
    config=None,
    block_size_k=64,
):
    """
    Fused all-gather + GEMM collective operation.

    Computes: output = all_gather(local_shard, dim=1) @ weight

    Each rank holds a column shard of the activation matrix. This operation
    gathers all shards via XGMI reads and fuses the communication with GEMM
    computation — as each rank's shard is read, its partial matmul contribution
    is immediately accumulated.

    Args:
        output_tensor: Output tensor of shape (M, N).
        local_shard: Local rank's column shard of shape (M, K_local).
        weight: Replicated weight matrix of shape (K, N) where K = world_size * K_local.
        shmem: Iris shmem context.
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If False, performs barrier at end. Default: False.
        config: Config instance with kernel parameters. Default: None (uses defaults).
        block_size_k: GEMM K-dimension block size. Default: 64.
    """
    if config is None:
        config = Config()

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    M, K_local = local_shard.shape
    K, N = weight.shape

    expected_K = world_size * K_local
    assert K == expected_K, (
        f"weight K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"
    )
    assert output_tensor.shape == (M, N), (
        f"output must be ({M}, {N}), got {output_tensor.shape}"
    )

    heap_bases = shmem.get_heap_bases()

    stride_am, stride_ak = local_shard.stride()
    stride_bk, stride_bn = weight.stride()
    stride_cm, stride_cn = output_tensor.stride()

    BLOCK_K = block_size_k
    even_k = K_local % BLOCK_K == 0
    num_k_blocks_local = (K_local + BLOCK_K - 1) // BLOCK_K

    _fused_ag_gemm_kernel[(config.comm_sms,)](
        local_shard,
        weight,
        output_tensor,
        M,
        N,
        K,
        K_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        heap_bases,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config.block_size_m,
        config.block_size_n,
        BLOCK_K,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_k_blocks_local,
        even_k,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
    )

    if not async_op:
        shmem.barrier()
