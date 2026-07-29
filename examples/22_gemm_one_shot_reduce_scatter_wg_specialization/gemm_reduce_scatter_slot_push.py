# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
GEMM + ReduceScatter with slot-based push (no element-wise atomics).

Instead of iris.atomic_add per element, each rank iris.store's its tile
to a per-source-rank slot on the owner. The owner then locally sums
all ws slots. Eliminates remote atomics entirely — bulk store only.

Memory: C_slots[ws, M_local, N] on each rank (in symmetric heap).
"""

import triton
import triton.language as tl
import iris


@triton.jit()
def persistent_gemm_reduce_scatter_slot_push(
    A,
    B,
    C_slots,
    C_out,
    gemm_locks,
    scatter_done,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_slot_rank,
    stride_slot_m,
    stride_slot_n,
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Three-phase fused GEMM+RS:
    1. GEMM WGs compute partial C, store to local buffer, set gemm_lock
    2. Comm WGs wait for gemm_lock, iris.store tile to owner's C_slots[source_rank]
       Set scatter_done flag on owner rank
    3. Reduce WGs (on owner) wait for all scatter_done[0..ws-1], sum slots → C_out
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    M_per_rank = M // world_size
    num_local_m_tiles = M_per_rank // BLOCK_SIZE_M
    total_local_tiles = num_local_m_tiles * num_pid_n

    acc_dtype = tl.float32

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    # Phase split: GEMM WGs | Comm WGs | Reduce WGs
    COMM_SMS: tl.constexpr = (NUM_SMS - GEMM_SMS) // 2
    REDUCE_SMS: tl.constexpr = NUM_SMS - GEMM_SMS - COMM_SMS

    if pid < GEMM_SMS:
        # ==========================================================
        # GEMM PHASE — compute partial C = A @ B, store locally
        # ==========================================================
        for tile_id in range(pid, total_tiles, GEMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            rk = tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            loop_k = tl.cdiv(K, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k -= 1

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for k in range(0, loop_k):
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
                acc += tl.dot(a, b)
                A_BASE += BLOCK_SIZE_K * stride_ak
                B_BASE += BLOCK_SIZE_K * stride_bk

            if not EVEN_K:
                k = loop_k
                rk2 = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_LAST = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
                B_LAST = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(A_LAST, mask=rk2[None, :] < K, other=0.0)
                b = tl.load(B_LAST, mask=rk2[:, None] < K, other=0.0)
                acc += tl.dot(a, b)

            c = acc.to(C_slots.type.element_ty)

            # Store to own slot on the OWNER rank's C_slots
            tile_m_start = pid_m * BLOCK_SIZE_M
            target_rank = tile_m_start // M_per_rank
            target_m = tile_m_start % M_per_rank
            offs_m = target_m + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # Write to C_slots[cur_rank, target_m:target_m+bm, pid_n*bn:pid_n*bn+bn]
            slot_offset = cur_rank * stride_slot_rank + offs_m[:, None] * stride_slot_m + offs_n[None, :] * stride_slot_n
            slot_mask = (offs_m[:, None] < M_per_rank) & (offs_n[None, :] < N)

            if target_rank == cur_rank:
                tl.store(C_slots + slot_offset, c, mask=slot_mask, cache_modifier=".wt")
            else:
                iris.store(
                    C_slots + slot_offset,
                    c,
                    cur_rank,
                    target_rank,
                    heap_bases,
                    mask=slot_mask,
                )

            # Signal: this tile's contribution from cur_rank is stored on target
            # scatter_done[source_rank * total_local_tiles + local_tile_id]
            local_pid_m = target_m // BLOCK_SIZE_M
            local_tile_id = local_pid_m * num_pid_n + pid_n
            flag_idx = cur_rank * total_local_tiles + local_tile_id

            if target_rank == cur_rank:
                tl.atomic_xchg(scatter_done + flag_idx, 1, sem="release", scope="gpu")
            else:
                iris.atomic_cas(
                    scatter_done + flag_idx, 0, 1,
                    cur_rank, target_rank, heap_bases,
                    sem="release", scope="sys",
                )

    elif pid < GEMM_SMS + COMM_SMS:
        # Comm WGs not needed in this variant — GEMM WGs push directly
        pass

    else:
        # ==========================================================
        # REDUCE PHASE — wait for all ranks' slots, sum → C_out
        # ==========================================================
        reduce_pid = pid - GEMM_SMS - COMM_SMS

        for local_tile_id in range(reduce_pid, total_local_tiles, REDUCE_SMS):
            local_pid_m = local_tile_id // num_pid_n
            pid_n = local_tile_id % num_pid_n

            # Wait for ALL ranks to push their contributions
            for src_rank in tl.static_range(world_size):
                flag_idx = src_rank * total_local_tiles + local_tile_id
                while tl.atomic_add(scatter_done + flag_idx, 0, sem="acquire", scope="sys") == 0:
                    pass

            # Sum all ws slots
            offs_m = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            slot_m_offset = offs_m[:, None] * stride_slot_m + offs_n[None, :] * stride_slot_n
            out_mask = (offs_m[:, None] < M_per_rank) & (offs_n[None, :] < N)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for src in tl.static_range(world_size):
                slot_ptr = C_slots + src * stride_slot_rank + slot_m_offset
                tile = tl.load(slot_ptr, mask=out_mask, other=0.0)
                acc += tile.to(acc_dtype)

            result = acc.to(C_out.type.element_ty)
            out_offset = offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
            tl.store(C_out + out_offset, result, mask=out_mask)
