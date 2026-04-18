# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-All KV transfer for fully-connected xGMI topologies (MI300X).

Design
------
On MI300X all 8 GPUs are connected via XGMI with uniform 1-hop latency to
every peer. The optimal KV distribution pattern is a direct all-to-all
scatter: each rank simultaneously iris.store's its local KV to ALL other
ranks' receive buffers, saturating all N*(N-1) = 56 xGMI links at once.

Two-kernel approach (fixes the single-kernel CTA synchronization bug)
----------------------------------------------------------------------
Single-kernel multi-CTA designs have a fundamental Triton limitation:
there is no grid-level barrier, so pid_tile==0 cannot signal the remote GPU
on behalf of other tiles that may not have finished their iris.stores yet.

Solution: split into two kernel launches. The CUDA kernel boundary provides
an implicit full-GPU barrier — when the host sees kernel 1 complete, ALL
CTAs (and therefore ALL iris.stores) are done.

  Kernel 1 — iris_alltoall_store_kernel   grid=(num_tiles, world_size-1)
    Each CTA stores one tile of local KV to each remote GPU's recv slot.
    (world_size-1)*num_tiles CTAs run simultaneously, saturating xGMI BW.

  Kernel 2 — iris_alltoall_signal_wait_kernel   grid=(world_size-1,)
    Called after kernel 1 completes. Signals all peers (1 atomic per dest)
    and spins until all peers have signalled back. Resets own flag.

Overlap variant (iris_alltoall_prefill_fused_kernel)
-----------------------------------------------------
Inspired by all_gather_matmul_hbm_buffer.py, which splits the grid into
dedicated FETCHER and GEMM workgroups within one kernel launch. Fetchers set
per-tile flags after each store; GEMM WGs spin on those flags and immediately
start computing, overlapping attention with in-flight transfers from other
source ranks.

Buffer layout on each GPU's sym-heap:
  k_recv_ptr[src * chunk_elems : (src+1)*chunk_elems] = KV sent by rank src
  tile_flags[src * num_tiles + tile_idx] = 1 when tile tile_idx from src ready
"""

import triton
import triton.language as tl
import iris
from iris.x.prefill_attn import flash_prefill_step


# ---------------------------------------------------------------------------
# Kernel 1: store tiles — grid=(num_tiles, world_size-1)
# ---------------------------------------------------------------------------

@triton.jit
def iris_alltoall_store_kernel(
    k_local_ptr,         # local K to send: flat [chunk_elems]
    v_local_ptr,         # local V to send: flat [chunk_elems]
    k_recv_ptr,          # sym-heap recv buf: [world_size * chunk_elems]
    v_recv_ptr,          # sym-heap recv buf: [world_size * chunk_elems]
    context_tensor,
    chunk_elems,         # S_local * H_kv * D
    num_tiles,           # cdiv(chunk_elems, BLOCK) — total tile count
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,  # fixed CTA count, each iterates many tiles
    BLOCK: tl.constexpr,
):
    """
    Persistent-style fetcher: grid=(NUM_FETCH_SMS * (world_size-1),).

    Inspired by all_gather_matmul_hbm_buffer.py: instead of one CTA per tile
    (which creates too many CTAs at large seq_len and becomes scheduler-bound),
    use a fixed small number of CTAs per destination. Each CTA loops over its
    assigned tile range in a stride pattern — persistent kernel style.

    pid_dest  — which remote destination (0..world_size-2 → actual rank)
    pid_fetch — which fetcher CTA for this destination (0..NUM_FETCH_SMS-1)

    Each (pid_dest, pid_fetch) CTA handles tiles:
      tile_idx = pid_fetch, pid_fetch + NUM_FETCH_SMS, pid_fetch + 2*NUM_FETCH_SMS, ...
    """
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    pid = tl.program_id(0)
    pid_dest  = pid // NUM_FETCH_SMS   # which destination rank
    pid_fetch = pid  % NUM_FETCH_SMS   # which fetcher within that destination

    # Map pid_dest to actual remote rank (skip cur_rank)
    dest_rank = pid_dest + (1 if pid_dest >= cur_rank else 0)

    # Slot for cur_rank on dest GPU
    dst_k = k_recv_ptr + cur_rank * chunk_elems
    dst_v = v_recv_ptr + cur_rank * chunk_elems

    # Each fetcher strides over its tiles
    for tile_idx in range(pid_fetch, num_tiles, NUM_FETCH_SMS):
        offs = tile_idx * BLOCK + tl.arange(0, BLOCK)
        mask = offs < chunk_elems

        k_data = tl.load(k_local_ptr + offs, mask=mask, other=0.0)
        v_data = tl.load(v_local_ptr + offs, mask=mask, other=0.0)

        # compile-time branch per destination rank
        # hint=BLOCK: tells Triton the pointer is BLOCK-aligned and contiguous,
        # enabling 128-bit vector stores instead of scalar buffer_store_short
        for r in tl.static_range(world_size):
            if r != cur_rank:
                if dest_rank == r:
                    iris.store(dst_k + offs, k_data, cur_rank, r, ctx.heap_bases, mask=mask, hint=BLOCK)
                    iris.store(dst_v + offs, v_data, cur_rank, r, ctx.heap_bases, mask=mask, hint=BLOCK)


# ---------------------------------------------------------------------------
# Kernel 2: signal and wait — grid=(world_size-1,)
# ---------------------------------------------------------------------------

@triton.jit
def iris_alltoall_signal_wait_kernel(
    ready_flags_ptr,     # sym-heap flags: [world_size] int32
    context_tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Grid: (world_size-1,).

    Called after iris_alltoall_store_kernel completes (kernel boundary = full
    GPU barrier). Each CTA signals one remote GPU then one designated CTA
    (pid==0) waits for all peers.

    By the time this kernel launches, all stores from kernel 1 are done and
    sys-scope visible, so no debug_barrier needed here.
    """
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

    pid = tl.program_id(0)
    dest_rank = pid + (1 if pid >= cur_rank else 0)

    # Signal: write ready_flags[cur_rank] = 1 on remote GPU dest_rank
    for r in tl.static_range(world_size):
        if r != cur_rank:
            if dest_rank == r:
                iris.atomic_xchg(
                    ready_flags_ptr + cur_rank, 1,
                    cur_rank, r, ctx.heap_bases,
                    sem="release", scope="sys",
                )

    # Designated CTA (pid==0) waits for all peers and resets own flag
    if pid == 0:
        for r in tl.static_range(world_size):
            if r != cur_rank:
                while tl.atomic_cas(
                    ready_flags_ptr + r, 0, 0, sem="acquire", scope="sys"
                ) == 0:
                    pass
        tl.atomic_xchg(ready_flags_ptr + cur_rank, 0, sem="release", scope="sys")


# ---------------------------------------------------------------------------
# Fused overlap variant: fetcher + attention workgroups in one kernel launch
# Inspired by all_gather_matmul_hbm_buffer.py
# ---------------------------------------------------------------------------

@triton.jit
def iris_alltoall_prefill_fused_kernel(
    # Send buffers (local HBM, flat [chunk_elems] = [S_local * H_kv * HEAD_DIM])
    k_local_ptr, v_local_ptr,
    # Receive buffers (sym-heap, [world_size * chunk_elems])
    # Layout: slot src_rank = k_recv_ptr + src_rank * chunk_elems
    k_recv_ptr, v_recv_ptr,
    # Per-tile-per-src flags (sym-heap, [world_size * num_kv_tiles])
    # tile_flags[src * num_kv_tiles + tile] = 1 when tile ready from src
    tile_flags_ptr,
    context_tensor,
    # Attention inputs/outputs
    q_ptr,       # [S_local, H, HEAD_DIM]
    out_ptr,     # [S_local, H, HEAD_DIM]
    # Q strides
    stride_qs, stride_qh, stride_qd,
    # Output strides
    stride_os, stride_oh, stride_od,
    # Metadata (runtime)
    S_local,
    H,
    chunk_elems,      # S_local * H_kv * HEAD_DIM
    num_kv_tiles,     # cdiv(S_local, BLOCK_K)
    scale,
    q_global_offset,  # cur_rank * S_local  (for causal masking)
    # Constexprs
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    H_kv: tl.constexpr,
    H_PER_KV: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,
    ATTN_PER_STAGE: tl.constexpr,  # attention CTAs per stage (co-scheduled with fetchers)
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TRANSFER: tl.constexpr,  # must divide BLOCK_K * H_kv * HEAD_DIM
):
    """
    Fused all-to-all KV scatter + causal prefill attention with compute/comm overlap.

    Inspired by all_gather_matmul_hbm_buffer.py.

    Grid: (NUM_FETCH_SMS + num_q_tiles * H,)

    The first NUM_FETCH_SMS CTAs are FETCHER workgroups:
      - Loop over their assigned KV tiles (stride pattern: tile, tile+NUM_FETCH_SMS, ...)
      - For each KV tile: iris.store to ALL remote recv buffers simultaneously
        (H_kv*HEAD_DIM elements per token × BLOCK_K tokens = KV_TILE_ELEMS elements)
      - Also copy to own local recv slot (regular tl.store)
      - After all stores for this tile: set per-tile flag on ALL remote GPUs
        (sys-scope release atomic) and on local GPU (gpu-scope)
      - Both local and remote attention CTAs spin on their respective flag

    Remaining CTAs are ATTENTION workgroups (one per q_tile × head):
      - Keep full online-softmax state (acc, e_max, e_sum) in registers
      - For each src_rank in 0..cur_rank (causal):
          For each kv_tile in 0..num_kv_tiles:
            Spin on tile_flags[src_rank * num_kv_tiles + kv_tile] (gpu-scope acquire)
            Load KV tile from recv buf (src_rank slot, strided [S_local, H_kv, D])
            Call flash_prefill_step to update softmax state
      - Finalize and write output directly (no HBM accumulator needed)

    Key memory ordering:
      Fetcher: iris.store (data) → tl.debug_barrier() →
               iris.atomic_xchg(flag=1, scope="sys") on remote GPU
               tl.atomic_xchg(flag=1, scope="gpu") on local GPU
      Attention: tl.atomic_add(flag, 0, scope="gpu") spin-wait  [reads sys-scope writes]
               → tl.load KV from recv buf
    """
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    pid = tl.program_id(0)

    # Number of flat elements per KV attention tile (BLOCK_K tokens, all heads+dims)
    KV_TILE_ELEMS: tl.constexpr = BLOCK_K * H_kv * HEAD_DIM
    # Number of BLOCK_TRANSFER-sized stores needed to send one KV attention tile
    INNER_TILES: tl.constexpr = KV_TILE_ELEMS // BLOCK_TRANSFER

    # Interleaved layout (inspired by all_gather_matmul_hbm_buffer.py):
    #   [fetch (NUM_FETCH_SMS)] [attn (ATTN_PER_STAGE)] [fetch (NUM_FETCH_SMS)] [attn ...]
    # This co-schedules fetcher and attention CTAs on the same XCDs so the
    # GPU scheduler runs them concurrently rather than sequentially.
    STAGE_SIZE: tl.constexpr = NUM_FETCH_SMS + ATTN_PER_STAGE

    stage = pid // STAGE_SIZE
    local_pid = pid % STAGE_SIZE

    if local_pid < NUM_FETCH_SMS:
        # =================================================================
        # FETCHER WORKGROUP
        # =================================================================
        fetch_pid = stage * NUM_FETCH_SMS + local_pid

        # This fetcher handles KV tiles: fetch_pid, fetch_pid+NF, fetch_pid+2*NF, ...
        for kv_tile in range(fetch_pid, num_kv_tiles, NUM_FETCH_SMS):
            # Flat byte-offset of this KV tile in k_local_ptr / k_recv_ptr
            kv_tile_base: tl.constexpr = kv_tile * KV_TILE_ELEMS  # runtime, not constexpr
            # Note: kv_tile is runtime so kv_tile_base is runtime too
            kv_tile_off = kv_tile * KV_TILE_ELEMS

            # --- Store to local recv slot (no RDMA) + all remote recv slots ---
            for inner in tl.static_range(INNER_TILES):
                offs = kv_tile_off + inner * BLOCK_TRANSFER + tl.arange(0, BLOCK_TRANSFER)
                mask = offs < chunk_elems

                k_data = tl.load(k_local_ptr + offs, mask=mask, other=0.0)
                v_data = tl.load(v_local_ptr + offs, mask=mask, other=0.0)

                # Local recv slot: regular HBM write (no RDMA)
                tl.store(k_recv_ptr + cur_rank * chunk_elems + offs, k_data, mask=mask)
                tl.store(v_recv_ptr + cur_rank * chunk_elems + offs, v_data, mask=mask)

                # Remote recv slots: iris.store to all other ranks simultaneously
                # Each store sends cur_rank's data to the slot reserved for cur_rank
                # on the remote GPU's recv buffer.
                # hint=BLOCK_TRANSFER enables 128-bit vector stores (see iris.x.all_gather)
                for r in tl.static_range(world_size):
                    if r != cur_rank:
                        dst_k = k_recv_ptr + cur_rank * chunk_elems
                        dst_v = v_recv_ptr + cur_rank * chunk_elems
                        iris.store(dst_k + offs, k_data, cur_rank, r,
                                   ctx.heap_bases, mask=mask, hint=BLOCK_TRANSFER)
                        iris.store(dst_v + offs, v_data, cur_rank, r,
                                   ctx.heap_bases, mask=mask, hint=BLOCK_TRANSFER)

            # Ensure all stores for this KV tile are ordered before signalling
            tl.debug_barrier()

            # Signal tile ready on all remote GPUs (sys-scope: visible cross-GPU)
            flag_idx = cur_rank * num_kv_tiles + kv_tile
            for r in tl.static_range(world_size):
                if r != cur_rank:
                    iris.atomic_xchg(
                        tile_flags_ptr + flag_idx, 1,
                        cur_rank, r, ctx.heap_bases,
                        sem="release", scope="sys",
                    )

            # Signal tile ready on local GPU (gpu-scope: visible to co-resident CTAs)
            tl.atomic_xchg(tile_flags_ptr + flag_idx, 1, sem="release", scope="gpu")

    else:
        # =================================================================
        # ATTENTION WORKGROUP: one CTA per (q_tile, query_head) pair
        # =================================================================
        attn_pid = stage * ATTN_PER_STAGE + (local_pid - NUM_FETCH_SMS)
        num_q_tiles = tl.cdiv(S_local, BLOCK_Q)

        pid_q = attn_pid % num_q_tiles
        pid_h = attn_pid // num_q_tiles

        if pid_h >= H:
            return  # out-of-bounds head

        kv_h = pid_h // H_PER_KV

        d_cols = tl.arange(0, HEAD_DIM)
        q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = (q_rows < S_local)[:, None]

        # Load Q tile once; keep in registers throughout
        q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
        q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

        # Online-softmax state: entirely in registers (no HBM round-trips)
        acc   = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
        e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

        q_global_off = q_global_offset + pid_q * BLOCK_Q

        # KV recv buf strides for layout [S_local, H_kv, HEAD_DIM]:
        #   stride_t = H_kv * HEAD_DIM,  stride_h = HEAD_DIM,  stride_d = 1
        stride_kbuf_t: tl.constexpr = H_kv * HEAD_DIM

        # Causal: attend to src_ranks 0..cur_rank (compile-time unrolled)
        for src_rank in tl.static_range(world_size):
            if src_rank <= cur_rank:
                kv_global_base = src_rank * S_local

                # KV tile loop — wait per tile so compute overlaps with in-flight sends
                for kv_tile in range(num_kv_tiles):
                    flag_idx = src_rank * num_kv_tiles + kv_tile

                    # Spin-wait: stall until fetcher sets this tile's flag
                    # tl.atomic_add(ptr, 0) = non-destructive read with acquire semantics
                    # gpu-scope: sufficient because fetcher CTAs are on the same GPU
                    # (sys-scope writes from remote fetchers are also visible to gpu-scope reads)
                    while tl.atomic_add(tile_flags_ptr + flag_idx, 0,
                                        sem="acquire", scope="gpu") == 0:
                        pass

                    # Load KV tile from recv buf (layout [S_local, H_kv, HEAD_DIM])
                    kv_start = kv_tile * BLOCK_K
                    kv_rows  = kv_start + tl.arange(0, BLOCK_K)
                    k_off = (src_rank * chunk_elems
                             + kv_rows[:, None] * stride_kbuf_t
                             + kv_h * HEAD_DIM
                             + d_cols[None, :])
                    k_tile = tl.load(k_recv_ptr + k_off).to(tl.float16)
                    v_tile = tl.load(v_recv_ptr + k_off).to(tl.float16)

                    acc, e_max, e_sum = flash_prefill_step(
                        q_tile, k_tile, v_tile,
                        acc, e_max, e_sum,
                        q_global_off,
                        kv_global_base + kv_start,
                        scale,
                        causal=(src_rank == cur_rank),
                        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
                    )

        # Finalize: normalize and write output directly (no separate finalize kernel)
        denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
        out   = (acc / denom).to(tl.float16)
        out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
        tl.store(out_ptr + out_off, out, mask=q_mask)


# ---------------------------------------------------------------------------
# DualChunkSwap fused kernel: same fetcher, DCS-aware attention CTA
# ---------------------------------------------------------------------------

@triton.jit
def iris_alltoall_prefill_dcs_kernel(
    # Send/recv buffers — identical layout to iris_alltoall_prefill_fused_kernel
    # Recv slot for src_rank: [pcp_tokens, H_kv, HEAD_DIM] in DCS order
    #   rows 0..chunk-1       = head chunk, global positions [src*chunk, (src+1)*chunk)
    #   rows chunk..2*chunk-1 = tail chunk, global positions [(2*ws-src-1)*chunk, (2*ws-src)*chunk)
    k_local_ptr, v_local_ptr,   # [chunk_elems] local KV, DCS order (head then tail)
    k_recv_ptr, v_recv_ptr,     # sym-heap [world_size * chunk_elems]
    tile_flags_ptr,             # sym-heap [world_size * num_kv_tiles]
    context_tensor,
    # Attention
    q_ptr,    # [pcp_tokens, H, HEAD_DIM] in DCS order: rows 0..chunk-1=HEAD, chunk..2chunk-1=TAIL
    out_ptr,  # [pcp_tokens, H, HEAD_DIM]
    stride_qs, stride_qh, stride_qd,
    stride_os, stride_oh, stride_od,
    # Runtime metadata
    pcp_tokens,               # 2 * chunk (tokens per rank)
    chunk,                    # padded // (2 * world_size)
    H,
    chunk_elems,              # pcp_tokens * H_kv * HEAD_DIM
    num_kv_tiles,             # pcp_tokens // BLOCK_K  (= 2 * head_tiles)
    scale,
    head_q_global_start,      # cur_rank * chunk
    tail_q_global_start,      # (2*ws - cur_rank - 1) * chunk
    # Constexprs
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    H_kv: tl.constexpr,
    H_PER_KV: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,
    ATTN_PER_STAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TRANSFER: tl.constexpr,
):
    """
    Fused all-to-all KV scatter + causal prefill attention for DualChunkSwap (DCS).

    FETCHER CTAs: identical to iris_alltoall_prefill_fused_kernel — push all pcp_tokens
    tiles (head+tail interleaved in DCS order) to all peers, signalling per-tile flags.

    ATTENTION CTAs: DCS-aware, no Q AllGather needed.
      Each rank's local Q is already in DCS order:
        rows 0..chunk-1       = HEAD tokens at global positions [cur_rank*chunk, (cur_rank+1)*chunk)
        rows chunk..2*chunk-1 = TAIL tokens at global positions [(2ws-cur_rank-1)*chunk, (2ws-cur_rank)*chunk)

      For HEAD Q tiles (pid_q * BLOCK_Q < chunk):
        q_global_off = head_q_global_start + pid_q * BLOCK_Q
        Need: head chunks of src_ranks 0..cur_rank only
        causal=(src_rank == cur_rank)  [compile-time, all other K tiles are strictly past]

      For TAIL Q tiles (pid_q * BLOCK_Q >= chunk):
        q_global_off = tail_q_global_start + (pid_q * BLOCK_Q - chunk)
        Need: head chunks of ALL src_ranks (all K pos < ws*chunk ≤ tail_q_global_start → causal=False)
            + tail chunks of src_ranks cur_rank..ws-1
        causal=False for head chunks, causal=(src_rank == cur_rank) for tail chunks

        Tail chunks of src_ranks 0..cur_rank-1 are SKIPPED: they are decode-only KV
        (their positions are beyond tail_q_global_start for this rank).

    Memory ordering: same as iris_alltoall_prefill_fused_kernel.
    """
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    pid = tl.program_id(0)

    KV_TILE_ELEMS: tl.constexpr = BLOCK_K * H_kv * HEAD_DIM
    INNER_TILES: tl.constexpr = KV_TILE_ELEMS // BLOCK_TRANSFER
    STAGE_SIZE: tl.constexpr = NUM_FETCH_SMS + ATTN_PER_STAGE

    stage = pid // STAGE_SIZE
    local_pid = pid % STAGE_SIZE

    if local_pid < NUM_FETCH_SMS:
        # =================================================================
        # FETCHER WORKGROUP — identical to iris_alltoall_prefill_fused_kernel
        # Pushes all num_kv_tiles tiles (head + tail of this rank's local KV)
        # to all remote recv bufs and sets per-tile flags.
        # =================================================================
        fetch_pid = stage * NUM_FETCH_SMS + local_pid

        for kv_tile in range(fetch_pid, num_kv_tiles, NUM_FETCH_SMS):
            kv_tile_off = kv_tile * KV_TILE_ELEMS

            for inner in tl.static_range(INNER_TILES):
                offs = kv_tile_off + inner * BLOCK_TRANSFER + tl.arange(0, BLOCK_TRANSFER)
                mask = offs < chunk_elems

                k_data = tl.load(k_local_ptr + offs, mask=mask, other=0.0)
                v_data = tl.load(v_local_ptr + offs, mask=mask, other=0.0)

                # Local recv slot
                tl.store(k_recv_ptr + cur_rank * chunk_elems + offs, k_data, mask=mask)
                tl.store(v_recv_ptr + cur_rank * chunk_elems + offs, v_data, mask=mask)

                # Remote recv slots
                for r in tl.static_range(world_size):
                    if r != cur_rank:
                        dst_k = k_recv_ptr + cur_rank * chunk_elems
                        dst_v = v_recv_ptr + cur_rank * chunk_elems
                        iris.store(dst_k + offs, k_data, cur_rank, r,
                                   ctx.heap_bases, mask=mask, hint=BLOCK_TRANSFER)
                        iris.store(dst_v + offs, v_data, cur_rank, r,
                                   ctx.heap_bases, mask=mask, hint=BLOCK_TRANSFER)

            tl.debug_barrier()

            flag_idx = cur_rank * num_kv_tiles + kv_tile
            for r in tl.static_range(world_size):
                if r != cur_rank:
                    iris.atomic_xchg(tile_flags_ptr + flag_idx, 1,
                                     cur_rank, r, ctx.heap_bases,
                                     sem="release", scope="sys")
            tl.atomic_xchg(tile_flags_ptr + flag_idx, 1, sem="release", scope="gpu")

    else:
        # =================================================================
        # ATTENTION WORKGROUP — DualChunkSwap-aware, selective KV gathering
        # =================================================================
        attn_pid = stage * ATTN_PER_STAGE + (local_pid - NUM_FETCH_SMS)
        num_q_tiles = tl.cdiv(pcp_tokens, BLOCK_Q)

        pid_q = attn_pid % num_q_tiles
        pid_h = attn_pid // num_q_tiles

        if pid_h >= H:
            return

        kv_h = pid_h // H_PER_KV
        d_cols = tl.arange(0, HEAD_DIM)
        q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = (q_rows < pcp_tokens)[:, None]

        q_off = q_rows[:, None] * stride_qs + pid_h * stride_qh + d_cols[None, :] * stride_qd
        q_tile = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float16)

        acc   = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
        e_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

        # head_tiles: tiles per chunk (runtime = num_kv_tiles // 2)
        head_tiles = num_kv_tiles // 2
        stride_kbuf_t: tl.constexpr = H_kv * HEAD_DIM

        # Determine if this Q tile is in the HEAD or TAIL region (runtime)
        q_local_start = pid_q * BLOCK_Q
        is_tail_q = q_local_start >= chunk

        # Compute global Q offset based on region (runtime branch)
        if is_tail_q:
            q_global_off = tail_q_global_start + q_local_start - chunk
        else:
            q_global_off = head_q_global_start + q_local_start

        # ------------------------------------------------------------------
        # Part A: Head chunks of src_ranks 0..cur_rank
        #
        # Always needed by both HEAD Q and TAIL Q.
        # For HEAD Q: causal=(src_rank==cur_rank) — only last head chunk overlaps.
        # For TAIL Q: all head K positions < ws*chunk ≤ tail_q_global_start,
        #             so k < q always → mask is all-True regardless of causal flag.
        #             Using causal=(src_rank==cur_rank) is correct for both cases.
        # ------------------------------------------------------------------
        for src_rank in tl.static_range(world_size):
            if src_rank <= cur_rank:
                for tile in range(head_tiles):
                    flag_idx = src_rank * num_kv_tiles + tile
                    while tl.atomic_add(tile_flags_ptr + flag_idx, 0,
                                        sem="acquire", scope="gpu") == 0:
                        pass

                    kv_rows = tile * BLOCK_K + tl.arange(0, BLOCK_K)
                    k_off = (src_rank * chunk_elems
                             + kv_rows[:, None] * stride_kbuf_t
                             + kv_h * HEAD_DIM
                             + d_cols[None, :])
                    k_tile = tl.load(k_recv_ptr + k_off).to(tl.float16)
                    v_tile = tl.load(v_recv_ptr + k_off).to(tl.float16)

                    acc, e_max, e_sum = flash_prefill_step(
                        q_tile, k_tile, v_tile, acc, e_max, e_sum,
                        q_global_off,
                        src_rank * chunk + tile * BLOCK_K,
                        scale,
                        causal=(src_rank == cur_rank),
                        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
                    )

        # ------------------------------------------------------------------
        # Part B: Head chunks of src_ranks cur_rank+1..ws-1 — TAIL Q only
        #
        # TAIL Q needs head chunks from ALL ws ranks (not just 0..cur_rank).
        # HEAD Q does not need these: their K positions > HEAD Q positions
        # would be all-masked anyway, but we skip the flag spin entirely.
        # causal=False: all head K positions < ws*chunk ≤ tail_q_global_start.
        # ------------------------------------------------------------------
        for src_rank in tl.static_range(world_size):
            if src_rank > cur_rank:
                if is_tail_q:
                    for tile in range(head_tiles):
                        flag_idx = src_rank * num_kv_tiles + tile
                        while tl.atomic_add(tile_flags_ptr + flag_idx, 0,
                                            sem="acquire", scope="gpu") == 0:
                            pass

                        kv_rows = tile * BLOCK_K + tl.arange(0, BLOCK_K)
                        k_off = (src_rank * chunk_elems
                                 + kv_rows[:, None] * stride_kbuf_t
                                 + kv_h * HEAD_DIM
                                 + d_cols[None, :])
                        k_tile = tl.load(k_recv_ptr + k_off).to(tl.float16)
                        v_tile = tl.load(v_recv_ptr + k_off).to(tl.float16)

                        acc, e_max, e_sum = flash_prefill_step(
                            q_tile, k_tile, v_tile, acc, e_max, e_sum,
                            q_global_off,
                            src_rank * chunk + tile * BLOCK_K,
                            scale,
                            causal=False,
                            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
                        )

        # ------------------------------------------------------------------
        # Part C: Tail chunks of src_ranks cur_rank..ws-1 — TAIL Q only
        #
        # Tail chunks of ranks 0..cur_rank-1 are SKIPPED (decode-only KV):
        #   Their global positions are [(2ws-src-1)*chunk, (2ws-src)*chunk)
        #   which are > tail_q_global_start for src < cur_rank, so they
        #   are entirely beyond what TAIL Q can attend to.
        # causal=(src_rank==cur_rank): only our own tail chunk overlaps with
        #   our TAIL Q tokens. All other tail chunks (src > cur_rank) have
        #   K positions < tail_q_global_start → causal=False.
        # ------------------------------------------------------------------
        for src_rank in tl.static_range(world_size):
            if src_rank >= cur_rank:
                if is_tail_q:
                    for tile in range(head_tiles):
                        flag_idx = src_rank * num_kv_tiles + head_tiles + tile
                        while tl.atomic_add(tile_flags_ptr + flag_idx, 0,
                                            sem="acquire", scope="gpu") == 0:
                            pass

                        # Tail chunk occupies rows chunk..2*chunk-1 in recv slot
                        kv_rows = chunk + tile * BLOCK_K + tl.arange(0, BLOCK_K)
                        k_off = (src_rank * chunk_elems
                                 + kv_rows[:, None] * stride_kbuf_t
                                 + kv_h * HEAD_DIM
                                 + d_cols[None, :])
                        k_tile = tl.load(k_recv_ptr + k_off).to(tl.float16)
                        v_tile = tl.load(v_recv_ptr + k_off).to(tl.float16)

                        # Tail chunk of src_rank: global pos = (2ws-src-1)*chunk + tile*BLOCK_K
                        acc, e_max, e_sum = flash_prefill_step(
                            q_tile, k_tile, v_tile, acc, e_max, e_sum,
                            q_global_off,
                            (2 * world_size - src_rank - 1) * chunk + tile * BLOCK_K,
                            scale,
                            causal=(src_rank == cur_rank),
                            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
                        )

        # Finalize
        denom = tl.where(e_sum[:, None] == 0.0, 1.0, e_sum[:, None])
        out_val = (acc / denom).to(tl.float16)
        out_off = q_rows[:, None] * stride_os + pid_h * stride_oh + d_cols[None, :] * stride_od
        tl.store(out_ptr + out_off, out_val, mask=q_mask)


# ---------------------------------------------------------------------------
# Host-side wrapper
# ---------------------------------------------------------------------------

def alltoall_prefill_dcs(
    k_local_flat,          # [chunk_elems] local K, DCS order (head then tail), HBM
    v_local_flat,          # [chunk_elems] local V, DCS order
    k_recv_buf,            # [world_size * chunk_elems] sym-heap recv buffer
    v_recv_buf,            # [world_size * chunk_elems] sym-heap recv buffer
    tile_flags,            # [world_size * num_kv_tiles] sym-heap per-tile flags
    q,                     # [pcp_tokens, H, D] query in DCS order
    out,                   # [pcp_tokens, H, D] output (pre-allocated)
    context_tensor,
    rank: int,
    world_size: int,
    pcp_tokens: int,       # tokens per rank = 2 * chunk
    chunk: int,            # padded // (2 * world_size)
    H: int,
    H_kv: int,
    scale: float,
    BLOCK_Q: int = 64,
    BLOCK_K: int = 64,
    BLOCK_TRANSFER: int = 256,
    NUM_FETCH_SMS: int = 32,
):
    """
    Fused DualChunkSwap all-to-all KV scatter + causal prefill attention.

    No Q AllGather needed. The kernel splits Q into HEAD and TAIL regions
    using the DCS layout (rows 0..chunk-1 = HEAD, chunk..2*chunk-1 = TAIL)
    and computes correct causal attention for each half by spinning only on
    the per-tile flags for KV tiles that are actually required:

      HEAD Q: head chunks of src_ranks 0..cur_rank only
      TAIL Q: head chunks of ALL src_ranks + tail chunks of src_ranks cur_rank..ws-1

    Tail chunks of src_ranks 0..cur_rank-1 are never waited on (decode-only,
    skipped on the critical path).
    """
    import torch
    HEAD_DIM = triton.next_power_of_2(q.shape[2])
    H_PER_KV = H // H_kv
    chunk_elems = k_local_flat.numel()
    num_kv_tiles = pcp_tokens // BLOCK_K      # = 2 * (chunk // BLOCK_K)
    head_q_global_start = rank * chunk
    tail_q_global_start = (2 * world_size - rank - 1) * chunk

    kv_tile_elems = BLOCK_K * H_kv * HEAD_DIM
    assert kv_tile_elems % BLOCK_TRANSFER == 0, (
        f"BLOCK_TRANSFER={BLOCK_TRANSFER} must divide BLOCK_K*H_kv*HEAD_DIM={kv_tile_elems}"
    )
    assert pcp_tokens % BLOCK_K == 0, "pcp_tokens must be divisible by BLOCK_K"
    assert chunk % BLOCK_K == 0, "chunk must be divisible by BLOCK_K (= head_tiles must be integer)"

    num_q_tiles = triton.cdiv(pcp_tokens, BLOCK_Q)
    num_attn_ctas = num_q_tiles * H

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    ATTN_PER_STAGE = max(1, num_sms - NUM_FETCH_SMS)
    num_stages = triton.cdiv(num_attn_ctas, ATTN_PER_STAGE)
    grid = (num_stages * (NUM_FETCH_SMS + ATTN_PER_STAGE),)

    iris_alltoall_prefill_dcs_kernel[grid](
        k_local_flat, v_local_flat,
        k_recv_buf, v_recv_buf,
        tile_flags,
        context_tensor,
        q, out,
        q.stride(0), q.stride(1), q.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        pcp_tokens, chunk, H, chunk_elems, num_kv_tiles, scale,
        head_q_global_start, tail_q_global_start,
        cur_rank=rank,
        world_size=world_size,
        H_kv=H_kv,
        H_PER_KV=H_PER_KV,
        NUM_FETCH_SMS=NUM_FETCH_SMS,
        ATTN_PER_STAGE=ATTN_PER_STAGE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=HEAD_DIM,
        BLOCK_TRANSFER=BLOCK_TRANSFER,
    )


def alltoall_prefill_fused(
    k_local_flat,    # [chunk_elems] local K (HBM, flat [S_local*H_kv*D])
    v_local_flat,    # [chunk_elems] local V (HBM)
    k_recv_buf,      # [world_size * chunk_elems] sym-heap recv buffer
    v_recv_buf,      # [world_size * chunk_elems] sym-heap recv buffer
    tile_flags,      # [world_size * num_kv_tiles] sym-heap per-tile flags
    q,               # [S_local, H, D] query tensor
    out,             # [S_local, H, D] output tensor (pre-allocated)
    context_tensor,
    rank: int,
    world_size: int,
    S_local: int,
    H: int,
    H_kv: int,
    scale: float,
    BLOCK_Q: int = 64,
    BLOCK_K: int = 64,
    BLOCK_TRANSFER: int = 256,
    NUM_FETCH_SMS: int = 32,
):
    """
    Fused all-to-all KV scatter + causal prefill attention with overlap.

    Launches a single kernel grid:
      (NUM_FETCH_SMS + cdiv(S_local, BLOCK_Q) * H,)

    First NUM_FETCH_SMS CTAs push local KV to all peers tile-by-tile, setting
    per-tile flags as they go. Remaining CTAs compute attention, spinning on
    per-tile flags and computing flash_prefill_step immediately as each tile
    arrives — overlapping compute with in-flight transfers.

    tile_flags must be zeroed before each call (preamble).
    """
    HEAD_DIM = triton.next_power_of_2(q.shape[2])
    H_PER_KV = H // H_kv
    chunk_elems = k_local_flat.numel()
    num_kv_tiles = triton.cdiv(S_local, BLOCK_K)
    q_global_offset = rank * S_local
    num_q_tiles = triton.cdiv(S_local, BLOCK_Q)
    num_attn_ctas = num_q_tiles * H

    # Validate BLOCK_TRANSFER divides KV_TILE_ELEMS
    kv_tile_elems = BLOCK_K * H_kv * HEAD_DIM
    assert kv_tile_elems % BLOCK_TRANSFER == 0, (
        f"BLOCK_TRANSFER={BLOCK_TRANSFER} must divide "
        f"BLOCK_K*H_kv*HEAD_DIM={kv_tile_elems}"
    )

    # Interleaved layout: alternate fetcher and attention stages so they
    # co-schedule on the same XCDs and genuinely overlap.
    # Each stage = NUM_FETCH_SMS fetchers + ATTN_PER_STAGE attention CTAs.
    # ATTN_PER_STAGE ~ 1 GPU wave = num_sms - NUM_FETCH_SMS.
    # This ensures fetchers and attention CTAs always land on the same wave.
    import torch
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    ATTN_PER_STAGE = max(1, num_sms - NUM_FETCH_SMS)  # ~272 for MI300X
    num_stages = triton.cdiv(num_attn_ctas, ATTN_PER_STAGE)
    grid = (num_stages * (NUM_FETCH_SMS + ATTN_PER_STAGE),)

    iris_alltoall_prefill_fused_kernel[grid](
        k_local_flat, v_local_flat,
        k_recv_buf, v_recv_buf,
        tile_flags,
        context_tensor,
        q, out,
        q.stride(0), q.stride(1), q.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        S_local, H, chunk_elems, num_kv_tiles, scale, q_global_offset,
        cur_rank=rank,
        world_size=world_size,
        H_kv=H_kv,
        H_PER_KV=H_PER_KV,
        NUM_FETCH_SMS=NUM_FETCH_SMS,
        ATTN_PER_STAGE=ATTN_PER_STAGE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=HEAD_DIM,
        BLOCK_TRANSFER=BLOCK_TRANSFER,
    )


def alltoall_kv(
    k_local_flat,         # [chunk_elems] local K (HBM)
    v_local_flat,         # [chunk_elems] local V (HBM)
    k_recv_buf,           # [world_size * chunk_elems] sym-heap recv buffer
    v_recv_buf,           # [world_size * chunk_elems] sym-heap recv buffer
    ready_flags,          # [world_size] sym-heap int32 flags
    context_tensor,
    rank: int,
    world_size: int,
    BLOCK: int = 1024,
    num_fetch_sms: int = 32,
):
    """
    Two-kernel all-to-all KV scatter with persistent-style fetchers.

    Kernel 1: iris_alltoall_store_kernel — grid=(num_fetch_sms*(world_size-1),)
      Each of the (world_size-1) destinations gets num_fetch_sms dedicated CTAs
      that loop over their tile range in a stride pattern. This avoids the
      scheduler overhead of one CTA per tile at large seq_len.

    Kernel 2: iris_alltoall_signal_wait_kernel — grid=(world_size-1,)
      After all stores complete (kernel boundary = full GPU barrier), signal
      all peers and wait for their acknowledgements.

    After this call, k_recv_buf[r*chunk_elems:(r+1)*chunk_elems] holds KV
    from rank r for all r in 0..world_size-1.
    """
    chunk_elems = k_local_flat.numel()
    num_tiles = triton.cdiv(chunk_elems, BLOCK)

    # Kernel 1: persistent fetchers — num_fetch_sms CTAs per destination
    iris_alltoall_store_kernel[(num_fetch_sms * (world_size - 1),)](
        k_local_flat, v_local_flat,
        k_recv_buf, v_recv_buf,
        context_tensor,
        chunk_elems,
        num_tiles,
        cur_rank=rank,
        world_size=world_size,
        NUM_FETCH_SMS=num_fetch_sms,
        BLOCK=BLOCK,
    )
    # Kernel boundary: implicit full-GPU barrier — all stores done

    # Kernel 2: signal peers and wait — (world_size-1) CTAs
    iris_alltoall_signal_wait_kernel[(world_size - 1,)](
        ready_flags,
        context_tensor,
        cur_rank=rank,
        world_size=world_size,
    )
