# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Ring Prefill Attention primitives for Iris.

This module implements device functions and kernels for prefill-phase (multi-token query)
attention in a distributed setting using ring communication.

Phases implemented:
  Phase 1 — flash_prefill_step, flash_prefill_kernel   (single GPU, non-paged)
  Phase 2 — load_kv_tile_paged, paged_prefill_attn_kernel  (single GPU, paged)
  Phase 4 — load_kv_tile_global_paged                      (multi-GPU, remote paged)
  Phase 5 — distributed_prefill_attn_kernel                (AllGather, multi-GPU)
  Phase 7 — ring_prefill_attn_step_kernel, finalize_prefill_output_kernel (unfused ring step)
  Phase 8 — fused_ring_prefill_attn_kernel                 (single persistent fused kernel)
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import iris
from iris.iris import DeviceContext


# ---------------------------------------------------------------------------
# Phase 1 — Reference single-GPU prefill attention (non-paged)
# ---------------------------------------------------------------------------


@triton.jit
def flash_prefill_step(
    q_tile,       # [BLOCK_Q, HEAD_DIM] fp16
    k_tile,       # [BLOCK_K, HEAD_DIM] fp16
    v_tile,       # [BLOCK_K, HEAD_DIM] fp16
    acc,          # [BLOCK_Q, HEAD_DIM] fp32 accumulated output
    e_max,        # [BLOCK_Q] fp32 running max logit
    e_sum,        # [BLOCK_Q] fp32 running denominator
    q_offset,     # int: global row index of q_tile[0] (for causal mask)
    kv_offset,    # int: global row index of k_tile[0] (for causal mask)
    scale,        # float: attention scale (1/sqrt(d))
    causal: tl.constexpr,   # bool constexpr: apply causal mask?
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Online-softmax flash attention inner-loop step.

    Computes QK^T for one (Q-tile, K-tile) pair, optionally applies causal mask,
    updates online softmax state (e_max, e_sum, acc), and returns the new state.

    Returns:
        (acc, e_max, e_sum): updated online softmax state
    """
    # QK^T: [BLOCK_Q, HEAD_DIM] x [HEAD_DIM, BLOCK_K] -> [BLOCK_Q, BLOCK_K]
    qk = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32)
    qk = qk * scale

    if causal:
        # Position j (key) can be attended by position i (query) only if j <= i
        q_idx = q_offset + tl.arange(0, BLOCK_Q)   # [BLOCK_Q]
        k_idx = kv_offset + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        causal_mask = q_idx[:, None] >= k_idx[None, :]  # [BLOCK_Q, BLOCK_K]
        qk = tl.where(causal_mask, qk, float("-inf"))

    # Online softmax update
    row_max = tl.max(qk, axis=1)               # [BLOCK_Q]
    n_e_max = tl.maximum(e_max, row_max)        # [BLOCK_Q]
    alpha = libdevice.fast_expf(e_max - n_e_max)  # [BLOCK_Q] rescale factor
    p = libdevice.fast_expf(qk - n_e_max[:, None])  # [BLOCK_Q, BLOCK_K] softmax numerator

    # Weighted value accumulation
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float32)
    e_sum = e_sum * alpha + tl.sum(p, axis=1)
    e_max = n_e_max

    return acc, e_max, e_sum


@triton.jit
def flash_prefill_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    S, H, H_kv,
    stride_qs, stride_qh, stride_qd,
    stride_ks, stride_kh, stride_kd,
    stride_vs, stride_vh, stride_vd,
    stride_os, stride_oh, stride_od,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    H_PER_KV: tl.constexpr,     # H // H_kv (GQA ratio)
    HEAD_DIM: tl.constexpr,
):
    """
    Single-GPU non-paged causal flash prefill attention kernel.

    Grid: (cdiv(S, BLOCK_Q), H)
    Each program handles one query tile × one query head.
    GQA: multiple query heads share one KV head (H_PER_KV query heads per KV head).
    """
    pid_q = tl.program_id(0)   # which BLOCK_Q slice of the sequence
    pid_h = tl.program_id(1)   # which query head

    # GQA: map query head to its KV head
    kv_h = pid_h // H_PER_KV

    # Query tile indices
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    d_cols = tl.arange(0, HEAD_DIM)                    # [HEAD_DIM]
    q_mask = (q_rows < S)[:, None]                     # [BLOCK_Q, 1]

    # Load Q tile: [BLOCK_Q, HEAD_DIM]
    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    # Online softmax state
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Causal: only attend to positions <= current Q tile's last position
    # so num_k_tiles = ceil((pid_q+1)*BLOCK_Q / BLOCK_K)
    num_k_tiles = tl.cdiv((pid_q + 1) * BLOCK_Q, BLOCK_K)

    for kv_block in range(0, num_k_tiles):
        kv_start = kv_block * BLOCK_K
        kv_rows = kv_start + tl.arange(0, BLOCK_K)
        kv_mask = (kv_rows < S)[:, None]  # [BLOCK_K, 1]

        # Load K tile: [BLOCK_K, HEAD_DIM]
        k_off = kv_rows[:, None] * stride_ks + kv_h * stride_kh + d_cols[None, :] * stride_kd
        k_tile = tl.load(k_ptr + k_off, mask=kv_mask, other=0.0).to(tl.float16)

        # Load V tile: [BLOCK_K, HEAD_DIM]
        v_off = kv_rows[:, None] * stride_vs + kv_h * stride_vh + d_cols[None, :] * stride_vd
        v_tile = tl.load(v_ptr + v_off, mask=kv_mask, other=0.0).to(tl.float16)

        q_offset = pid_q * BLOCK_Q
        acc, e_max, e_sum = flash_prefill_step(
            q_tile, k_tile, v_tile,
            acc, e_max, e_sum,
            q_offset, kv_start,
            scale, causal=True,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
        )

    # Normalize and write output
    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)


def reference_prefill_attn(q, k, v, scale=None, causal=True):
    """
    Host wrapper for single-GPU non-paged flash prefill attention.

    Args:
        q: [S, H, D] query tensor (fp16, contiguous)
        k: [S, H_kv, D] key tensor (fp16, contiguous)
        v: [S, H_kv, D] value tensor (fp16, contiguous)
        scale: attention scale (default: D^{-0.5})
        causal: apply causal mask (default True; non-causal not tested)

    Returns:
        out: [S, H, D] attention output (fp16)
    """
    S, H, D = q.shape
    H_kv = k.shape[1]
    assert H % H_kv == 0, f"H={H} must be divisible by H_kv={H_kv}"
    H_PER_KV = H // H_kv

    if scale is None:
        scale = D ** -0.5

    # Tile sizes
    BLOCK_Q = min(64, triton.next_power_of_2(S))
    BLOCK_K = min(64, triton.next_power_of_2(S))
    HEAD_DIM = triton.next_power_of_2(D)

    # Pad head dim to power of 2
    if HEAD_DIM != D:
        q = torch.nn.functional.pad(q, (0, HEAD_DIM - D))
        k = torch.nn.functional.pad(k, (0, HEAD_DIM - D))
        v = torch.nn.functional.pad(v, (0, HEAD_DIM - D))

    out = torch.empty(S, H, HEAD_DIM, dtype=torch.float16, device=q.device)

    grid = (triton.cdiv(S, BLOCK_Q), H)
    flash_prefill_kernel[grid](
        q, k, v, out,
        S, H, H_kv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        H_PER_KV=H_PER_KV, HEAD_DIM=HEAD_DIM,
    )

    if HEAD_DIM != D:
        out = out[:, :, :D]
    return out.contiguous()


# ---------------------------------------------------------------------------
# Phase 2 — Paged prefill attention (single GPU)
# ---------------------------------------------------------------------------


@triton.jit
def paged_prefill_attn_kernel(
    q_ptr,
    k_cache_ptr, v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    # Q strides: q shape [S, H, D]
    stride_qs, stride_qh, stride_qd,
    # KV cache strides: cache shape [num_phys_blocks, PAGE_SIZE, H_kv, D]
    stride_cb, stride_cs, stride_ch, stride_cd,
    # block_table strides: [batch, max_blocks]
    stride_tb, stride_tn,
    # output strides: [S, H, D]
    stride_os, stride_oh, stride_od,
    # metadata
    S, H, H_kv,
    scale,
    H_PER_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Single-GPU paged prefill attention kernel.

    KV cache has shape [num_phys_blocks, PAGE_SIZE, H_kv, D].
    block_table has shape [batch, max_blocks]: block_table[b, i] = physical block index
    for logical KV block i of sequence b.

    Grid: (batch, cdiv(S, BLOCK_Q), H)
    """
    bid = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    kv_h = pid_h // H_PER_KV

    kv_len = tl.load(seq_lens_ptr + bid)

    # Query tile
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, HEAD_DIM)
    q_mask = (q_rows < S)[:, None]

    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Causal: only attend up to q_tile's last position or kv_len, whichever is smaller
    max_kv = tl.minimum(kv_len, (pid_q + 1) * BLOCK_Q)
    num_k_tiles = tl.cdiv(max_kv, BLOCK_K)

    for kv_block in range(0, num_k_tiles):
        kv_start = kv_block * BLOCK_K
        kv_rows = kv_start + tl.arange(0, BLOCK_K)

        # Block table lookup: physical block for logical block (kv_rows // PAGE_SIZE)
        phys_block = tl.load(
            block_table_ptr + bid * stride_tb + (kv_rows // PAGE_SIZE) * stride_tn,
        )  # [BLOCK_K]
        slot = kv_rows % PAGE_SIZE  # [BLOCK_K]

        # Physical KV cache offset for this KV head
        # cache[phys_block, slot, kv_h, d] = cache_ptr + phys_block*stride_cb + slot*stride_cs + kv_h*stride_ch + d*stride_cd
        k_off = (
            phys_block[:, None] * stride_cb
            + slot[:, None] * stride_cs
            + kv_h * stride_ch
            + d_cols[None, :] * stride_cd
        )  # [BLOCK_K, HEAD_DIM]

        k_tile = tl.load(k_cache_ptr + k_off).to(tl.float16)
        v_tile = tl.load(v_cache_ptr + k_off).to(tl.float16)  # v_cache has same layout

        q_offset = pid_q * BLOCK_Q
        acc, e_max, e_sum = flash_prefill_step(
            q_tile, k_tile, v_tile,
            acc, e_max, e_sum,
            q_offset, kv_start,
            scale, causal=True,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
        )

    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)


def paged_prefill_attn(q, k_cache, v_cache, block_table, seq_lens, scale=None):
    """
    Host wrapper for single-GPU paged prefill attention.

    Args:
        q: [S, H, D] query tensor (fp16)
        k_cache: [num_blocks, PAGE_SIZE, H_kv, D] key cache (fp16)
        v_cache: [num_blocks, PAGE_SIZE, H_kv, D] value cache (fp16)
        block_table: [batch, max_blocks] int32 physical block indices
        seq_lens: [batch] int32 actual KV sequence lengths
        scale: attention scale (default 1/sqrt(D))

    Returns:
        out: [S, H, D] output tensor (fp16)
    """
    S, H, D = q.shape
    _, PAGE_SIZE, H_kv, _ = k_cache.shape
    assert H % H_kv == 0
    H_PER_KV = H // H_kv
    batch = block_table.shape[0]

    if scale is None:
        scale = D ** -0.5

    BLOCK_Q = min(64, triton.next_power_of_2(S))
    BLOCK_K = PAGE_SIZE  # align BLOCK_K with page size for clean block table lookup
    HEAD_DIM = triton.next_power_of_2(D)

    if HEAD_DIM != D:
        q = torch.nn.functional.pad(q, (0, HEAD_DIM - D))

    out = torch.empty(S, H, HEAD_DIM, dtype=torch.float16, device=q.device)

    grid = (batch, triton.cdiv(S, BLOCK_Q), H)
    paged_prefill_attn_kernel[grid](
        q, k_cache, v_cache, block_table, seq_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        S, H, H_kv,
        scale,
        H_PER_KV=H_PER_KV, PAGE_SIZE=PAGE_SIZE,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
    )

    if HEAD_DIM != D:
        out = out[:, :, :D]
    return out.contiguous()


# ---------------------------------------------------------------------------
# Phase 4 — load_kv_tile_global_paged (multi-GPU, remote block table access)
# ---------------------------------------------------------------------------


@triton.jit
def load_kv_tile_global_paged(
    k_cache_ptr, v_cache_ptr,
    global_block_table_ptr,
    heap_bases,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    bid,
    kv_start,
    kv_h,
    # KV cache strides (same on all ranks)
    stride_cb, stride_cs, stride_ch, stride_cd,
    # global_block_table strides: [batch, max_global_blocks, 2]
    stride_gbt_b, stride_gbt_blk,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Load a K and V tile from the global paged KV cache (supports cross-rank access).

    The global_block_table has shape [batch, max_global_blocks, 2]:
      [..., 0] = owning rank
      [..., 1] = local physical block index on that rank

    For local blocks: uses tl.load.
    For remote blocks: uses iris.load with heap_bases for pointer translation.

    Assumes BLOCK_K <= PAGE_SIZE so that all positions in the tile belong to the
    same physical page (and thus the same owning rank).

    Returns:
        k_tile [BLOCK_K, HEAD_DIM], v_tile [BLOCK_K, HEAD_DIM] (fp16)
    """
    kv_rows = kv_start + tl.arange(0, BLOCK_K)   # [BLOCK_K] logical positions
    d_cols = tl.arange(0, HEAD_DIM)               # [HEAD_DIM]

    # Global block index and slot within page
    global_blk_idx = kv_rows // PAGE_SIZE          # [BLOCK_K]
    slot = kv_rows % PAGE_SIZE                     # [BLOCK_K]

    # Load (owning_rank, local_phys_block) from global block table
    gbt_base = bid * stride_gbt_b + global_blk_idx * stride_gbt_blk
    owning_rank = tl.load(global_block_table_ptr + gbt_base + 0)   # [BLOCK_K]
    local_phys_block = tl.load(global_block_table_ptr + gbt_base + 1)  # [BLOCK_K]

    # Physical offset within the owning rank's KV cache
    kv_off = (
        local_phys_block[:, None] * stride_cb
        + slot[:, None] * stride_cs
        + kv_h * stride_ch
        + d_cols[None, :] * stride_cd
    )  # [BLOCK_K, HEAD_DIM]

    # owning_rank is a runtime tensor; we compare against my_rank to branch local vs remote.
    # Since all positions in the tile share the same page (BLOCK_K <= PAGE_SIZE),
    # owning_rank is uniform. We use a scalar comparison trick: sum and divide by BLOCK_K.
    # Alternatively, load just the first element by reducing to a scalar.
    owner_scalar = tl.sum(owning_rank, axis=0) // BLOCK_K  # uniform, so this gives the rank

    # Branch: local load if owner == my_rank, else remote iris.load.
    # We must iterate over all ranks at compile time to generate constexpr from_rank/to_rank.
    k_tile = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float16)
    v_tile = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float16)

    for r in range(0, world_size):
        if r == my_rank:
            # Local path: check at runtime if this rank owns the block
            if owner_scalar == my_rank:
                k_tile = tl.load(k_cache_ptr + kv_off)
                v_tile = tl.load(v_cache_ptr + kv_off)
        else:
            # Remote path: check at runtime if rank r owns the block
            if owner_scalar == r:
                k_tile = iris.load(k_cache_ptr + kv_off, my_rank, r, heap_bases)
                v_tile = iris.load(v_cache_ptr + kv_off, my_rank, r, heap_bases)

    return k_tile.to(tl.float16), v_tile.to(tl.float16)


# ---------------------------------------------------------------------------
# Phase 5 — AllGather-based distributed prefill (non-pipelined, multi-GPU)
# ---------------------------------------------------------------------------


@triton.jit
def distributed_prefill_attn_kernel(
    q_ptr,
    k_cache_ptr, v_cache_ptr,
    global_block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    context_tensor,
    # Q strides: [S_local, H, D]
    stride_qs, stride_qh, stride_qd,
    # KV cache strides: [num_phys_blocks, PAGE_SIZE, H_kv, D]
    stride_cb, stride_cs, stride_ch, stride_cd,
    # global_block_table strides: [batch, max_global_blocks, 2]
    stride_gbt_b, stride_gbt_blk,
    # output strides: [S_local, H, D]
    stride_os, stride_oh, stride_od,
    # metadata
    S_local, H, H_kv,
    chunk_len,   # tokens per rank (S_local when all ranks have same length)
    scale,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    H_PER_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Distributed prefill attention: each rank processes its local Q chunk, pulling
    KV from all ranks via the global block table (iris.load for remote, tl.load local).

    Causal masking: rank r attends only to KV chunks from ranks 0..r (past and self).
    The KV chunk for rank kv_rank covers global positions [kv_rank*chunk_len, (kv_rank+1)*chunk_len).

    Grid: (batch, cdiv(S_local, BLOCK_Q), H)
    """
    bid = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_h = tl.program_id(2)

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    kv_h = pid_h // H_PER_KV

    # Query tile
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, HEAD_DIM)
    q_mask = (q_rows < S_local)[:, None]

    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Global query offset for causal mask
    q_global_offset = cur_rank * chunk_len + pid_q * BLOCK_Q

    # Loop over all source ranks (causal: skip future ranks)
    for kv_rank in range(0, world_size):
        if kv_rank <= cur_rank:
            kv_global_base = kv_rank * chunk_len
            num_kv_tiles = tl.cdiv(chunk_len, BLOCK_K)

            for kv_block in range(0, num_kv_tiles):
                kv_start_global = kv_global_base + kv_block * BLOCK_K

                # Look up physical block via global block table
                kv_rows_g = kv_start_global + tl.arange(0, BLOCK_K)
                global_blk_idx = kv_rows_g // PAGE_SIZE
                slot = kv_rows_g % PAGE_SIZE

                # The owning rank for blocks in this kv_rank's chunk IS kv_rank
                # (guaranteed by build_global_block_table layout).
                # Load the local physical block index from the global block table.
                gbt_base = bid * stride_gbt_b + global_blk_idx * stride_gbt_blk
                local_phys_block = tl.load(global_block_table_ptr + gbt_base + 1)

                kv_off = (
                    local_phys_block[:, None] * stride_cb
                    + slot[:, None] * stride_cs
                    + kv_h * stride_ch
                    + d_cols[None, :] * stride_cd
                )

                # kv_rank is constexpr (loop var over compile-time range), so this if is compile-time
                if kv_rank == cur_rank:
                    k_tile = tl.load(k_cache_ptr + kv_off)
                    v_tile = tl.load(v_cache_ptr + kv_off)
                else:
                    k_tile = iris.load(k_cache_ptr + kv_off, cur_rank, kv_rank, ctx.heap_bases)
                    v_tile = iris.load(v_cache_ptr + kv_off, cur_rank, kv_rank, ctx.heap_bases)

                is_causal = (kv_rank == cur_rank)
                acc, e_max, e_sum = flash_prefill_step(
                    q_tile, k_tile.to(tl.float16), v_tile.to(tl.float16),
                    acc, e_max, e_sum,
                    q_global_offset, kv_start_global,
                    scale,
                    causal=is_causal,
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
                )

    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)


# ---------------------------------------------------------------------------
# Phase 7 — Unfused ring prefill attention (per-step kernel)
# ---------------------------------------------------------------------------


@triton.jit
def ring_prefill_attn_step_kernel(
    q_ptr,
    k_recv_buf_ptr, v_recv_buf_ptr,
    acc_ptr, e_max_ptr, e_sum_ptr,
    # strides for Q: [S_local, H, D]
    stride_qs, stride_qh, stride_qd,
    # strides for ring KV buffer: [chunk_len, H_kv, D]
    stride_kbuf_t, stride_kbuf_h, stride_kbuf_d,
    # strides for acc: [S_local, H, D]
    stride_acc_t, stride_acc_h, stride_acc_d,
    # strides for e_max/e_sum: [S_local, H]
    stride_e_t, stride_e_h,
    # metadata
    S_local, H, H_kv,
    chunk_len,
    q_global_offset,    # cur_rank * chunk_len
    kv_global_offset,   # kv_rank * chunk_len
    scale,
    kv_rank: tl.constexpr,
    cur_rank: tl.constexpr,
    H_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    One step of unfused ring prefill attention.

    Loads Q, reads acc/e_max/e_sum from HBM, attends against the KV in the ring
    receive buffer (already transferred by the host), updates and writes back state.

    Grid: (cdiv(S_local, BLOCK_Q), H)
    """
    # Skip future positions (causal)
    if kv_rank > cur_rank:
        return

    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    kv_h = pid_h // H_PER_KV
    is_causal = (kv_rank == cur_rank)

    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, HEAD_DIM)
    q_mask = (q_rows < S_local)[:, None]

    # Load Q tile
    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    # Load persistent softmax state from HBM
    acc_off = q_rows[:, None] * stride_acc_t + pid_h * stride_acc_h + d_cols[None, :] * stride_acc_d
    e_off = q_rows * stride_e_t + pid_h * stride_e_h

    acc = tl.load(acc_ptr + acc_off, mask=q_mask, other=0.0).to(tl.float32)
    e_max = tl.load(e_max_ptr + e_off, mask=(q_rows < S_local), other=float("-inf")).to(tl.float32)
    e_sum = tl.load(e_sum_ptr + e_off, mask=(q_rows < S_local), other=0.0).to(tl.float32)

    # Attend to all KV blocks in the ring buffer
    num_kv_tiles = tl.cdiv(chunk_len, BLOCK_K)
    for kv_block in range(0, num_kv_tiles):
        kv_start = kv_block * BLOCK_K
        kv_rows = kv_start + tl.arange(0, BLOCK_K)

        k_off = kv_rows[:, None] * stride_kbuf_t + kv_h * stride_kbuf_h + d_cols[None, :] * stride_kbuf_d
        k_tile = tl.load(k_recv_buf_ptr + k_off).to(tl.float16)
        v_tile = tl.load(v_recv_buf_ptr + k_off).to(tl.float16)

        acc, e_max, e_sum = flash_prefill_step(
            q_tile, k_tile, v_tile,
            acc, e_max, e_sum,
            q_global_offset + pid_q * BLOCK_Q,
            kv_global_offset + kv_start,
            scale,
            causal=is_causal,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
        )

    # Write back updated softmax state
    tl.store(acc_ptr + acc_off, acc.to(tl.float16), mask=q_mask)
    tl.store(e_max_ptr + e_off, e_max, mask=(q_rows < S_local))
    tl.store(e_sum_ptr + e_off, e_sum, mask=(q_rows < S_local))


@triton.jit
def finalize_prefill_output_kernel(
    acc_ptr, e_max_ptr, e_sum_ptr, out_ptr,
    stride_acc_t, stride_acc_h, stride_acc_d,
    stride_e_t, stride_e_h,
    stride_os, stride_oh, stride_od,
    S_local, H,
    BLOCK_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Normalize accumulated attention state and write final output."""
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, HEAD_DIM)
    q_mask = (q_rows < S_local)[:, None]

    acc_off = q_rows[:, None] * stride_acc_t + pid_h * stride_acc_h + d_cols[None, :] * stride_acc_d
    e_off = q_rows * stride_e_t + pid_h * stride_e_h

    acc = tl.load(acc_ptr + acc_off, mask=q_mask, other=0.0).to(tl.float32)
    e_sum = tl.load(e_sum_ptr + e_off, mask=(q_rows < S_local), other=1.0).to(tl.float32)

    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)


# ---------------------------------------------------------------------------
# Phase 8 — Fused ring prefill attention (single persistent kernel)
# ---------------------------------------------------------------------------


@triton.jit
def fused_ring_prefill_attn_kernel(
    q_ptr,
    k_local_ptr, v_local_ptr,       # local KV on sym heap: [chunk_len, H_kv, HEAD_DIM]
    k_ring_A_ptr, v_ring_A_ptr,     # ring buffer A (sym heap): [chunk_len * H_kv * HEAD_DIM]
    k_ring_B_ptr, v_ring_B_ptr,     # ring buffer B (sym heap)
    signal_flags_A_ptr,             # sym heap flags [H_kv]: one flag per KV head for buf A
    signal_flags_B_ptr,             # sym heap flags [H_kv]: one flag per KV head for buf B
    out_ptr,
    context_tensor,
    # Q strides: [S_local, H, D]
    stride_qs, stride_qh, stride_qd,
    # local KV strides: [chunk_len, H_kv, HEAD_DIM]
    stride_lkv_t, stride_lkv_h, stride_lkv_d,
    # ring buffer strides: [chunk_len, H_kv, HEAD_DIM] (flat layout)
    stride_rkv_t, stride_rkv_h, stride_rkv_d,
    # output strides: [S_local, H, D]
    stride_os, stride_oh, stride_od,
    # metadata
    S_local, H, H_kv,
    chunk_len,
    scale,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    H_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Fused ring prefill attention in a single Triton kernel per (Q-tile, head).

    Communication pattern (world_size=N):
      Step 0: Use local KV, send local KV to next rank via ring buffer A.
      Step s (1..N-1):
        - Spin-wait on ring buffer (A if s odd, B if s even)
        - If not last step: relay received KV to next rank (using opposite buffer)
        - If kv_rank <= cur_rank: compute attention
        - Reset flag

    This kernel fuses compute and communication: the relay store to next rank
    is initiated while the local attention computation proceeds.

    Grid: (cdiv(S_local, BLOCK_Q), H)
    """
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    next_rank = (cur_rank + 1) % world_size

    kv_h = pid_h // H_PER_KV

    # Query tile
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, HEAD_DIM)
    q_mask = (q_rows < S_local)[:, None]

    q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
    q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

    # Online softmax state
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    q_global_base = cur_rank * chunk_len
    num_kv_tiles = tl.cdiv(chunk_len, BLOCK_K)

    # --------------- Step 0: local KV attention + start ring send ---------------
    # One CTA per KV head handles the send (pid_q == 0 as the designated sender)
    if pid_q == 0:
        # Push local KV chunk to next rank's ring buffer A
        for kv_block_s in range(0, num_kv_tiles):
            kv_rows_s = kv_block_s * BLOCK_K + tl.arange(0, BLOCK_K)
            k_loc_off_s = kv_rows_s[:, None] * stride_lkv_t + kv_h * stride_lkv_h + d_cols[None, :] * stride_lkv_d
            k_s = tl.load(k_local_ptr + k_loc_off_s)
            v_s = tl.load(v_local_ptr + k_loc_off_s)

            r_off_s = kv_rows_s[:, None] * stride_rkv_t + kv_h * stride_rkv_h + d_cols[None, :] * stride_rkv_d
            iris.store(k_ring_A_ptr + r_off_s, k_s, cur_rank, next_rank, ctx.heap_bases)
            iris.store(v_ring_A_ptr + r_off_s, v_s, cur_rank, next_rank, ctx.heap_bases)

        tl.debug_barrier()
        # Signal next rank: buffer A slot kv_h is ready
        iris.atomic_xchg(signal_flags_A_ptr + kv_h, 1, cur_rank, next_rank, ctx.heap_bases,
                         sem="release", scope="sys")

    tl.debug_barrier()

    # Local attention (kv_rank == cur_rank, causal)
    for kv_block in range(0, num_kv_tiles):
        kv_start = kv_block * BLOCK_K
        kv_rows = kv_start + tl.arange(0, BLOCK_K)
        k_loc_off = kv_rows[:, None] * stride_lkv_t + kv_h * stride_lkv_h + d_cols[None, :] * stride_lkv_d
        k_tile = tl.load(k_local_ptr + k_loc_off).to(tl.float16)
        v_tile = tl.load(v_local_ptr + k_loc_off).to(tl.float16)

        acc, e_max, e_sum = flash_prefill_step(
            q_tile, k_tile, v_tile,
            acc, e_max, e_sum,
            q_global_base + pid_q * BLOCK_Q,
            q_global_base + kv_start,  # kv_rank == cur_rank
            scale, causal=True,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
        )

    # --------------- Steps 1..world_size-1: ring recv + optional relay + attention ---------------
    for step in range(1, world_size):
        kv_rank = (cur_rank - step + world_size) % world_size

        # Which buffer to receive from (double buffered: A for odd, B for even)
        use_A = (step % 2 == 1)

        if use_A:
            recv_k_ptr = k_ring_A_ptr
            recv_v_ptr = v_ring_A_ptr
            recv_flags = signal_flags_A_ptr
            send_k_ptr = k_ring_B_ptr
            send_v_ptr = v_ring_B_ptr
            send_flags = signal_flags_B_ptr
        else:
            recv_k_ptr = k_ring_B_ptr
            recv_v_ptr = v_ring_B_ptr
            recv_flags = signal_flags_B_ptr
            send_k_ptr = k_ring_A_ptr
            send_v_ptr = v_ring_A_ptr
            send_flags = signal_flags_A_ptr

        # Spin-wait for incoming data (flag set by previous rank's send)
        while tl.atomic_cas(recv_flags + kv_h, 0, 0, sem="acquire", scope="sys") == 0:
            pass

        # Relay to next rank (overlap with attention compute)
        if step < world_size - 1:
            if pid_q == 0:
                for kv_block_r in range(0, num_kv_tiles):
                    kv_rows_r = kv_block_r * BLOCK_K + tl.arange(0, BLOCK_K)
                    r_off_r = kv_rows_r[:, None] * stride_rkv_t + kv_h * stride_rkv_h + d_cols[None, :] * stride_rkv_d
                    k_relay = tl.load(recv_k_ptr + r_off_r)
                    v_relay = tl.load(recv_v_ptr + r_off_r)
                    iris.store(send_k_ptr + r_off_r, k_relay, cur_rank, next_rank, ctx.heap_bases)
                    iris.store(send_v_ptr + r_off_r, v_relay, cur_rank, next_rank, ctx.heap_bases)

                tl.debug_barrier()
                iris.atomic_xchg(send_flags + kv_h, 1, cur_rank, next_rank, ctx.heap_bases,
                                 sem="release", scope="sys")

        # Reset consumed flag
        tl.atomic_xchg(recv_flags + kv_h, 0, sem="release", scope="sys")

        # Attend to received KV (causal: skip kv_rank > cur_rank)
        if kv_rank <= cur_rank:
            kv_global_base = kv_rank * chunk_len
            for kv_block in range(0, num_kv_tiles):
                kv_start = kv_block * BLOCK_K
                kv_rows = kv_start + tl.arange(0, BLOCK_K)
                r_off = kv_rows[:, None] * stride_rkv_t + kv_h * stride_rkv_h + d_cols[None, :] * stride_rkv_d
                k_tile = tl.load(recv_k_ptr + r_off).to(tl.float16)
                v_tile = tl.load(recv_v_ptr + r_off).to(tl.float16)

                acc, e_max, e_sum = flash_prefill_step(
                    q_tile, k_tile, v_tile,
                    acc, e_max, e_sum,
                    q_global_base + pid_q * BLOCK_Q,
                    kv_global_base + kv_start,
                    scale,
                    causal=(kv_rank == cur_rank),
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
                )

    # Normalize and store output
    denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
    out = (acc / denom).to(tl.float16)
    out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
    tl.store(out_ptr + out_off, out, mask=q_mask)
