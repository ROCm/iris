################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Ring Attention implementation based on:
#   "Ring Attention with Blockwise Transformers for Near-Infinite Context"
#   Liu et al., 2023 (https://arxiv.org/pdf/2310.01889)
#
################################################################################

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import iris


@triton.jit
def _ring_attn_persistent_kernel(
    Q,
    K_ping,
    K_pong,
    V_ping,
    V_pong,
    O,
    M_acc,
    L_acc,
    # flat pointers for iris.put (contiguous views of ping/pong buffers)
    K_ping_flat,
    K_pong_flat,
    V_ping_flat,
    V_pong_flat,
    # strides for Q, O: [seq, num_heads, head_dim]
    stride_qs,
    stride_qh,
    stride_qd,
    stride_os,
    stride_oh,
    stride_od,
    # strides for K_ping, V_ping (same for pong): [seq, num_heads, head_dim]
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    # strides for M_acc, L_acc: [num_heads, seq]
    stride_mh,
    stride_ms,
    stride_lh,
    stride_ls,
    # sizes
    seq_q,
    seq_kv,
    num_heads,
    # signal infrastructure
    signal_flags,
    put_done_counters,
    heap_bases,
    scale,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PUT_BLOCK: tl.constexpr,
):
    """
    Persistent flash attention kernel for ring attention.

    Runs the entire ring loop inside a single kernel launch. Q stays in
    registers across all steps. M, L, O accumulators stay in registers —
    no HBM round-trip between steps.

    Synchronization uses point-to-point signal flags on the symmetric heap
    instead of host-side barriers. Each CTA atomically increments a completion
    counter after its put; the last CTA fires a remote signal to the next rank.
    """
    h = tl.program_id(0)
    q_blk = tl.program_id(1)

    q_off = q_blk * BLOCK_Q
    q_idx = q_off + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < seq_q

    # Load Q once — stays in registers across all ring steps
    q_ptrs = Q + h * stride_qh + q_idx[:, None] * stride_qs + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Initialize accumulators in registers (no HBM round-trip between steps)
    m = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # Global Q positions for causal masking
    q_rank_start = rank * seq_q
    q_global = q_rank_start + q_idx
    q_global_max = q_rank_start + q_off + BLOCK_Q - 1

    num_q_blks = tl.cdiv(seq_q, BLOCK_Q)
    total_blocks = num_heads * num_q_blks
    next_rank = (rank + 1) % world_size
    n_put_elem = seq_kv * num_heads * HEAD_DIM

    d_idx = tl.arange(0, HEAD_DIM)

    for step in range(world_size):
        kv_rank = (rank - step) % world_size
        kv_rank_start = kv_rank * seq_kv
        do_put = step < world_size - 1

        # Select ping/pong buffer based on step parity
        # Structured pointers for attention loads, flat pointers for iris.put
        if step % 2 == 0:
            K_cur = K_ping
            V_cur = V_ping
            K_cur_flat = K_ping_flat
            V_cur_flat = V_ping_flat
            K_dst_flat = K_pong_flat
            V_dst_flat = V_pong_flat
        else:
            K_cur = K_pong
            V_cur = V_pong
            K_cur_flat = K_pong_flat
            V_cur_flat = V_pong_flat
            K_dst_flat = K_ping_flat
            V_dst_flat = V_ping_flat

        # WAIT: if step > 0, spin on signal from previous rank
        if step > 0:
            while tl.atomic_cas(signal_flags + step, 0, 0, sem="acquire", scope="sys") != step:
                pass

        # COMPUTE: flash attention on this KV chunk
        # Uses the same structure as the original kernel: iterate over KV blocks
        # with causal skip logic inside the loop body.
        for kv_off in range(0, seq_kv, BLOCK_KV):
            if CAUSAL:
                do_kv_block = kv_rank_start + kv_off <= q_global_max
            else:
                do_kv_block = True

            if do_kv_block:
                kv_idx = kv_off + tl.arange(0, BLOCK_KV)
                kv_mask = kv_idx < seq_kv

                k_ptrs = K_cur + h * stride_kh + d_idx[:, None] * stride_kd + kv_idx[None, :] * stride_ks
                v_ptrs = V_cur + h * stride_vh + kv_idx[:, None] * stride_vs + d_idx[None, :] * stride_vd

                k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

                qk = tl.dot(q, k) * scale

                if CAUSAL:
                    kv_global = kv_rank_start + kv_idx
                    causal_mask = kv_global[None, :] <= q_global[:, None]
                    qk = tl.where(causal_mask & kv_mask[None, :], qk, -float("inf"))
                else:
                    qk = tl.where(kv_mask[None, :], qk, -float("inf"))

                m_new = tl.maximum(m, tl.max(qk, axis=1))
                alpha = libdevice.fast_expf(m - m_new)
                p = libdevice.fast_expf(qk - m_new[:, None])
                l = alpha * l + tl.sum(p, axis=1)
                o = alpha[:, None] * o + tl.dot(p.to(v.dtype), v)
                m = m_new

        # COMMUNICATE: fused put to next rank's dst buffer
        if do_put:
            pid_flat = h * num_q_blks + q_blk
            put_offs = pid_flat * PUT_BLOCK + tl.arange(0, PUT_BLOCK)
            put_mask = put_offs < n_put_elem
            iris.put(K_cur_flat + put_offs, K_dst_flat + put_offs, rank, next_rank, heap_bases, mask=put_mask)
            iris.put(V_cur_flat + put_offs, V_dst_flat + put_offs, rank, next_rank, heap_bases, mask=put_mask)
            tl.debug_barrier()

            # Count completed CTAs; last one signals next rank.
            # scope="sys" on the counter ensures each CTA's remote puts are
            # visible system-wide before the counter increment is observed.
            old = tl.atomic_add(put_done_counters + step, 1, sem="release", scope="sys")
            if old == total_blocks - 1:
                iris.atomic_xchg(
                    signal_flags + step + 1, step + 1, rank, next_rank, heap_bases, sem="release", scope="sys"
                )

    # Store final O, M, L to HBM (once, not per-step)
    o_ptrs = O + h * stride_oh + q_idx[:, None] * stride_os + tl.arange(0, HEAD_DIM)[None, :] * stride_od
    m_ptrs = M_acc + h * stride_mh + q_idx * stride_ms
    l_ptrs = L_acc + h * stride_lh + q_idx * stride_ls
    tl.store(o_ptrs, o, mask=q_mask[:, None])
    tl.store(m_ptrs, m, mask=q_mask)
    tl.store(l_ptrs, l, mask=q_mask)


def ring_attn_fwd(q, k, v, shmem, causal=True, scale=None, _ping_pong_bufs=None, _signal_flags=None):
    """
    Ring Attention forward pass.

    Each device holds a contiguous chunk of the sequence (Q, K, V). K and V
    are rotated around the ring of devices using Iris ``put`` operations
    fused into a persistent kernel, while Q remains local. The entire ring
    loop runs inside a single kernel launch with point-to-point signal-flag
    synchronization — no host-side barriers between steps.

    After all ``world_size`` steps, O is normalised by L to produce the output.

    Args:
        q (torch.Tensor): Query tensor, shape ``[seq_q, num_heads, head_dim]``.
            Lives on the local device's CUDA memory.
        k (torch.Tensor): Key tensor, same shape as ``q``.
        v (torch.Tensor): Value tensor, same shape as ``q``.
        shmem: Iris shmem context (provides ``get_rank()`` / ``get_num_ranks()``,
            ``get_heap_bases()`` and ``barrier()``).
        causal (bool): If ``True``, apply a causal (lower-triangular) mask so
            that position ``i`` only attends to positions ``j <= i``.
        scale (float | None): Softmax scale factor. Defaults to
            ``head_dim ** -0.5``.
        _ping_pong_bufs (tuple | None): Optional pre-allocated ping-pong buffers
            ``(k_ping, k_pong, v_ping, v_pong)`` from the symmetric heap.
        _signal_flags (torch.Tensor | None): Optional pre-allocated signal flags
            on the symmetric heap, shape ``[world_size]``, dtype ``int32``.

    Returns:
        torch.Tensor: Attention output, shape ``[seq_q, num_heads, head_dim]``,
            same dtype as ``q``.
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[0]

    assert (head_dim & (head_dim - 1)) == 0, f"head_dim must be a power of 2, got {head_dim}"
    assert seq_q % 64 == 0, f"seq_q ({seq_q}) must be divisible by BLOCK_Q (64)"
    assert seq_kv % 64 == 0, f"seq_kv ({seq_kv}) must be divisible by BLOCK_KV (64)"

    if scale is None:
        scale = head_dim**-0.5

    input_dtype = q.dtype

    # Running accumulators in float32 for numerical stability
    O = torch.zeros(seq_q, num_heads, head_dim, dtype=torch.float32, device=q.device)
    M = torch.full((num_heads, seq_q), fill_value=-float("inf"), dtype=torch.float32, device=q.device)
    L = torch.zeros(num_heads, seq_q, dtype=torch.float32, device=q.device)

    BLOCK_Q = 64
    BLOCK_KV = 64
    HEAD_DIM = head_dim

    # Allocate ping-pong buffers on symmetric heap
    if _ping_pong_bufs is not None:
        k_ping, k_pong, v_ping, v_pong = _ping_pong_bufs
    else:
        k_ping = shmem.empty(k.shape, dtype=k.dtype)
        k_pong = shmem.empty(k.shape, dtype=k.dtype)
        v_ping = shmem.empty(v.shape, dtype=v.dtype)
        v_pong = shmem.empty(v.shape, dtype=v.dtype)

    # Allocate signal flags on symmetric heap (one per step, indexed 1..world_size-1)
    # and local completion counters (one per step)
    if _signal_flags is not None:
        signal_flags = _signal_flags
    else:
        signal_flags = shmem.zeros((world_size,), dtype=torch.int32)
    # Reset signal flags to 0 for this call
    signal_flags.zero_()
    put_done_counters = torch.zeros(world_size, dtype=torch.int32, device=q.device)

    # Copy initial K/V into ping buffers, then sync so every rank has its
    # own initial chunk ready before the persistent kernel launches.
    k_ping.copy_(k.contiguous())
    v_ping.copy_(v.contiguous())
    shmem.barrier()  # only host barrier — ensures all ranks have initial data

    FUSED_PUT_BLOCK = BLOCK_Q * HEAD_DIM
    heap_bases = shmem.get_heap_bases()

    # Single kernel launch for ALL ring steps
    grid = (num_heads, triton.cdiv(seq_q, BLOCK_Q))
    _ring_attn_persistent_kernel[grid](
        q,
        k_ping,
        k_pong,
        v_ping,
        v_pong,
        O,
        M,
        L,
        # flat pointers for iris.put
        k_ping.view(-1),
        k_pong.view(-1),
        v_ping.view(-1),
        v_pong.view(-1),
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # O strides
        O.stride(0),
        O.stride(1),
        O.stride(2),
        # K strides (ping and pong have same strides)
        k_ping.stride(0),
        k_ping.stride(1),
        k_ping.stride(2),
        # V strides
        v_ping.stride(0),
        v_ping.stride(1),
        v_ping.stride(2),
        # M, L strides
        M.stride(0),
        M.stride(1),
        L.stride(0),
        L.stride(1),
        # sizes
        seq_q,
        seq_kv,
        num_heads,
        # signal infrastructure
        signal_flags,
        put_done_counters,
        heap_bases,
        scale,
        rank=rank,
        world_size=world_size,
        CAUSAL=causal,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        HEAD_DIM=HEAD_DIM,
        PUT_BLOCK=FUSED_PUT_BLOCK,
        num_warps=4,
        num_stages=2,
    )

    # Normalize: output = O / L, where L is the softmax denominator
    L_expanded = L.permute(1, 0).unsqueeze(-1)  # [seq_q, num_heads, 1]
    output = O / L_expanded

    return output.to(input_dtype)
