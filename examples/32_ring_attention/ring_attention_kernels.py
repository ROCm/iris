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
def _put_kv_kernel(
    k_src,
    k_dst,
    v_src,
    v_dst,
    n_elem,
    cur_rank: tl.constexpr,
    next_rank: tl.constexpr,
    heap_bases,
    BLOCK: tl.constexpr,
):
    """
    Fused K+V put: copy K and V to the next rank in a single kernel launch.

    Both K and V tensors must be flat (same number of elements) and reside on
    the Iris symmetric heap so that their addresses can be translated to
    ``next_rank``'s address space.

    Each program instance copies ``BLOCK`` elements of K **and** ``BLOCK``
    elements of V, halving kernel-launch overhead compared to two separate
    ``_put_tensor_kernel`` calls.

    Args:
        k_src: Source K pointer (must be on the symmetric heap).
        k_dst: Destination K pointer (must be on the symmetric heap).
        v_src: Source V pointer (must be on the symmetric heap).
        v_dst: Destination V pointer (must be on the symmetric heap).
        n_elem: Total number of elements in K (same as V).
        cur_rank: This rank's ID.
        next_rank: Destination rank ID.
        heap_bases: Iris heap base address table.
        BLOCK: Number of elements each program instance handles.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    iris.put(k_src + offs, k_dst + offs, cur_rank, next_rank, heap_bases, mask=mask)
    iris.put(v_src + offs, v_dst + offs, cur_rank, next_rank, heap_bases, mask=mask)


@triton.jit
def _ring_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    M,
    L,
    # strides for Q, K, V, O: [seq, num_heads, head_dim]
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_os,
    stride_oh,
    stride_od,
    # strides for M, L: [num_heads, seq]
    stride_mh,
    stride_ms,
    stride_lh,
    stride_ls,
    # sizes
    seq_q,
    seq_kv,
    # global offsets for causal masking
    q_rank_start,
    kv_rank_start,
    scale,
    # fused KV put parameters
    k_put_src,
    k_put_dst,
    v_put_src,
    v_put_dst,
    n_put_elem,
    put_rank: tl.constexpr,
    put_next_rank: tl.constexpr,
    heap_bases,
    # compile-time constants
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DO_PUT: tl.constexpr,
    PUT_BLOCK: tl.constexpr,
):
    """
    Flash attention kernel for one ring step.

    Each program instance handles one attention head and one block of Q tokens.
    Iterates over all KV blocks and accumulates using online softmax.

    Accumulates into O (unnormalized), M (running log-sum-exp), L (running sum).
    The final output is O / L, computed after all ring steps complete.
    """
    h = tl.program_id(0)
    q_blk = tl.program_id(1)

    q_off = q_blk * BLOCK_Q
    q_idx = q_off + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < seq_q

    # Load Q block: [BLOCK_Q, HEAD_DIM]
    q_ptrs = Q + h * stride_qh + q_idx[:, None] * stride_qs + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Load running statistics for this head and Q block
    m_ptrs = M + h * stride_mh + q_idx * stride_ms
    l_ptrs = L + h * stride_lh + q_idx * stride_ls
    o_ptrs = O + h * stride_oh + q_idx[:, None] * stride_os + tl.arange(0, HEAD_DIM)[None, :] * stride_od

    m = tl.load(m_ptrs, mask=q_mask, other=-float("inf"))
    l = tl.load(l_ptrs, mask=q_mask, other=0.0)
    o = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Global Q positions for causal masking.
    # Triton loads q_rank_start (a Python int) and q_idx (int32 arange) as int32.
    # The maximum value is world_size * seq_q which fits comfortably in int32.
    q_global = q_rank_start + q_idx

    # Iterate over all KV blocks
    d_idx = tl.arange(0, HEAD_DIM)
    for kv_off in range(0, seq_kv, BLOCK_KV):
        kv_idx = kv_off + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < seq_kv

        # Load K in transposed layout [HEAD_DIM, BLOCK_KV] for efficient dot product
        k_ptrs = K + h * stride_kh + d_idx[:, None] * stride_kd + kv_idx[None, :] * stride_ks
        v_ptrs = V + h * stride_vh + kv_idx[:, None] * stride_vs + d_idx[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Attention scores: [BLOCK_Q, BLOCK_KV] = Q [BLOCK_Q, HEAD_DIM] @ K^T [HEAD_DIM, BLOCK_KV]
        qk = tl.dot(q, k) * scale

        # Apply padding mask (validity) and optional causal mask
        if CAUSAL:
            kv_global = kv_rank_start + kv_idx
            # Causal: token at kv_pos can only be attended to if kv_pos <= q_pos
            causal_mask = kv_global[None, :] <= q_global[:, None]
            qk = tl.where(causal_mask & kv_mask[None, :], qk, -float("inf"))
        else:
            qk = tl.where(kv_mask[None, :], qk, -float("inf"))

        # Online softmax accumulation
        # m_new = max(m, row_max(qk))
        m_new = tl.maximum(m, tl.max(qk, axis=1))

        # Scale factor for previous running values
        alpha = libdevice.fast_expf(m - m_new)

        # Unnormalized attention probabilities
        p = libdevice.fast_expf(qk - m_new[:, None])

        # Update running sum
        l = alpha * l + tl.sum(p, axis=1)

        # Update running output (unnormalized weighted value sum)
        o = alpha[:, None] * o + tl.dot(p, v)

        # Update running max
        m = m_new

    # Write back updated statistics and output
    tl.store(m_ptrs, m, mask=q_mask)
    tl.store(l_ptrs, l, mask=q_mask)
    tl.store(o_ptrs, o, mask=q_mask[:, None])

    # --- Fused KV rotation: each thread block transfers a slice of K and V ---
    # The attention grid has (num_heads * cdiv(seq_q, BLOCK_Q)) blocks.
    # Each block transfers PUT_BLOCK = BLOCK_Q * HEAD_DIM elements, so the
    # total coverage = num_heads * seq_q * HEAD_DIM = n_put_elem exactly.
    if DO_PUT:
        num_q_blks = tl.cdiv(seq_q, BLOCK_Q)
        pid_flat = h * num_q_blks + q_blk
        put_offs = pid_flat * PUT_BLOCK + tl.arange(0, PUT_BLOCK)
        put_mask = put_offs < n_put_elem
        iris.put(k_put_src + put_offs, k_put_dst + put_offs,
                 put_rank, put_next_rank, heap_bases, mask=put_mask)
        iris.put(v_put_src + put_offs, v_put_dst + put_offs,
                 put_rank, put_next_rank, heap_bases, mask=put_mask)


def ring_attn_fwd(q, k, v, shmem, causal=True, scale=None, _ping_pong_bufs=None):
    """
    Ring Attention forward pass.

    Each device holds a contiguous chunk of the sequence (Q, K, V). K and V
    are rotated around the ring of devices using Iris ``put`` operations (via
    ``_put_kv_kernel``), while Q remains local. At each step the local
    flash-attention kernel accumulates partial results into O, M, L using
    online softmax.

    After all ``world_size`` steps, O is normalised by L to produce the output.

    Communication uses two ping-pong symmetric buffers per tensor (K and V),
    allocated on the Iris heap.  After each push, ``shmem.barrier()`` ensures
    all ranks have received the new data before proceeding to the next step.

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
            ``(k_ping, k_pong, v_ping, v_pong)`` from the symmetric heap.  When
            provided, no new heap allocation is performed (avoids heap churn on
            repeated calls with the same tensor shape).

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
    # O is the *unnormalized* weighted value sum
    O = torch.zeros(seq_q, num_heads, head_dim, dtype=torch.float32, device=q.device)
    # M: running row-max (log domain), L: running normalisation denominator
    M = torch.full((num_heads, seq_q), fill_value=-float("inf"), dtype=torch.float32, device=q.device)
    L = torch.zeros(num_heads, seq_q, dtype=torch.float32, device=q.device)

    # Choose block sizes; keep them as powers of 2
    BLOCK_Q = 64
    BLOCK_KV = 64
    HEAD_DIM = head_dim  # already validated as power of 2

    # Allocate two symmetric ping-pong buffers per tensor on the Iris heap.
    # The destination buffer of each iris.put must be on the symmetric heap so
    # that the pointer can be translated to the remote rank's address space.
    # If the caller supplies pre-allocated buffers (e.g. from RingAttention),
    # reuse them to avoid heap churn on repeated forward passes.
    if _ping_pong_bufs is not None:
        k_ping, k_pong, v_ping, v_pong = _ping_pong_bufs
    else:
        k_ping = shmem.empty(k.shape, dtype=k.dtype)
        k_pong = shmem.empty(k.shape, dtype=k.dtype)
        v_ping = shmem.empty(v.shape, dtype=v.dtype)
        v_pong = shmem.empty(v.shape, dtype=v.dtype)

    # Copy initial K/V into the ping buffers, then sync so every rank has its
    # own initial chunk ready before the first rotation.
    k_ping.copy_(k.contiguous())
    v_ping.copy_(v.contiguous())
    shmem.barrier()

    k_cur, k_recv = k_ping, k_pong
    v_cur, v_recv = v_ping, v_pong

    next_rank = (rank + 1) % world_size

    STANDALONE_PUT_BLOCK = 1024
    FUSED_PUT_BLOCK = BLOCK_Q * HEAD_DIM
    n_k = k_cur.numel()
    heap_bases = shmem.get_heap_bases()

    for step in range(world_size):
        kv_rank = (rank - step) % world_size
        do_put = step < world_size - 1

        if causal:
            skip_compute = kv_rank > rank
            apply_causal = kv_rank == rank
        else:
            skip_compute = False
            apply_causal = False

        if not skip_compute:
            # Fused attention + KV rotation: the attention kernel's thread
            # blocks each transfer a slice of K/V to the next rank after
            # computing their attention tile, achieving SM-level overlap
            # with zero extra kernel launches.
            q_rank_start = rank * seq_q
            kv_rank_start = kv_rank * seq_kv
            grid = (num_heads, triton.cdiv(seq_q, BLOCK_Q))
            _ring_attn_fwd_kernel[grid](
                q,
                k_cur,
                v_cur,
                O,
                M,
                L,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k_cur.stride(0),
                k_cur.stride(1),
                k_cur.stride(2),
                v_cur.stride(0),
                v_cur.stride(1),
                v_cur.stride(2),
                O.stride(0),
                O.stride(1),
                O.stride(2),
                M.stride(0),
                M.stride(1),
                L.stride(0),
                L.stride(1),
                seq_q,
                seq_kv,
                q_rank_start,
                kv_rank_start,
                scale,
                # fused put params
                k_cur.view(-1),
                k_recv.view(-1),
                v_cur.view(-1),
                v_recv.view(-1),
                n_k,
                put_rank=rank,
                put_next_rank=next_rank,
                heap_bases=heap_bases,
                CAUSAL=apply_causal,
                BLOCK_Q=BLOCK_Q,
                BLOCK_KV=BLOCK_KV,
                HEAD_DIM=HEAD_DIM,
                DO_PUT=do_put,
                PUT_BLOCK=FUSED_PUT_BLOCK,
                num_warps=4,
                num_stages=2,
            )
        elif do_put:
            # skip_compute but still need to rotate KV — use standalone put
            _put_kv_kernel[(triton.cdiv(n_k, STANDALONE_PUT_BLOCK),)](
                k_cur.view(-1),
                k_recv.view(-1),
                v_cur.view(-1),
                v_recv.view(-1),
                n_k,
                cur_rank=rank,
                next_rank=next_rank,
                heap_bases=heap_bases,
                BLOCK=STANDALONE_PUT_BLOCK,
            )

        # Global barrier ensures all ranks have received data, then swap.
        if do_put:
            torch.cuda.synchronize()
            shmem.barrier()
            k_cur, k_recv = k_recv, k_cur
            v_cur, v_recv = v_recv, v_cur

    # Normalize: output = O / L, where L is the softmax denominator
    # L: [num_heads, seq_q] → [seq_q, num_heads, 1] for broadcasting
    L_expanded = L.permute(1, 0).unsqueeze(-1)  # [seq_q, num_heads, 1]
    output = O / L_expanded

    return output.to(input_dtype)
