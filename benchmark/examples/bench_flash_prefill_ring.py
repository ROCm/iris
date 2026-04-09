#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Ablation benchmark for distributed prefill attention strategies.

Compares 6 implementations across model shapes (LLaMA-3-8B/70B) and
sequence lengths (8K–256K) on 8× MI300X GPUs:

  local_flash       — compute ceiling, S_local tokens, all H heads, no comms
  tp                — head-parallel TP, S_total tokens, H/8 heads, no attn comms
  cp_allgather_nccl — dist.all_gather KV then local flash attn
  cp_allgather_iris — iris.load per KV block fused into attention loop
  cp_ring_unfused   — iris ring transfer kernel + PyTorch attention per step
  cp_ring_fused     — Phase 12 persistent fused kernel (skipped until built)

Metrics reported:
  gpu_time_ms  — wall-clock latency
  tflops       — from set_flops(4 * S_total^2 * H * D) for scaling efficiency
  comm_gb      — cross-GPU bytes / 1e9
  tok_per_sec  — S_total / latency

Run:
  python -m iris.bench benchmark/examples/bench_flash_prefill_ring.py
  python -m iris.bench benchmark/examples/bench_flash_prefill_ring.py \\
      --benchmark_format=csv > ablation_results.csv
  python -m iris.bench benchmark/examples/bench_flash_prefill_ring.py \\
      --axis_seq_len_total=8192,256000
"""

import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

import iris
import iris.bench as bench

# ---------------------------------------------------------------------------
# Project root + example path setup
# ---------------------------------------------------------------------------

_project_root = Path(__file__).resolve()
while not (_project_root / "tests").is_dir() or not (_project_root / "examples").is_dir():
    if _project_root == _project_root.parent:
        raise FileNotFoundError("Could not find project root")
    _project_root = _project_root.parent

_module_dir = _project_root / "examples" / "14_flash_prefill_ring"
if _module_dir.is_dir():
    sys.path.insert(0, str(_module_dir))

# ---------------------------------------------------------------------------
# Model shape registry
# ---------------------------------------------------------------------------

MODEL_SHAPES = {
    "llama3-8b":  (32, 8, 128),   # H=32, H_kv=8,  D=128  (GQA 4:1)
    "llama3-70b": (64, 8, 128),   # H=64, H_kv=8,  D=128  (GQA 8:1)
}

# ---------------------------------------------------------------------------
# Single-CTA iris ring transfer kernel (validated in Phase 8/9)
# ---------------------------------------------------------------------------


@triton.jit
def _iris_ring_transfer_kernel(
    send_k_ptr, send_v_ptr,
    recv_k_ptr, recv_v_ptr,
    signal_flags_ptr,
    chunk_elems,
    context_tensor,
    kv_h,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-CTA: send local KV slice to next rank via iris, receive from prev rank."""
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    next_rank = (cur_rank + 1) % world_size
    num_tiles = tl.cdiv(chunk_elems, BLOCK)
    for i in range(0, num_tiles):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < chunk_elems
        k_data = tl.load(send_k_ptr + offs, mask=mask, other=0.0)
        v_data = tl.load(send_v_ptr + offs, mask=mask, other=0.0)
        iris.store(recv_k_ptr + offs, k_data, cur_rank, next_rank, ctx.heap_bases, mask=mask)
        iris.store(recv_v_ptr + offs, v_data, cur_rank, next_rank, ctx.heap_bases, mask=mask)
    tl.debug_barrier()
    iris.atomic_xchg(signal_flags_ptr + kv_h, 1, cur_rank, next_rank, ctx.heap_bases,
                     sem="release", scope="sys")
    while tl.atomic_cas(signal_flags_ptr + kv_h, 0, 0, sem="acquire", scope="sys") == 0:
        pass
    tl.atomic_xchg(signal_flags_ptr + kv_h, 0, sem="release", scope="sys")


# ---------------------------------------------------------------------------
# Helper: SDPA with GQA expansion (handles asymmetric S_q / S_kv)
# ---------------------------------------------------------------------------


def _sdpa_attn(q, k, v, scale):
    """
    Scaled dot-product attention with GQA head expansion.

    Args:
        q: [S_q, H, D]  — query tensor
        k: [S_kv, H_kv, D] — key tensor
        v: [S_kv, H_kv, D] — value tensor
        scale: attention scale factor

    Returns:
        out: [S_q, H, D]
    """
    H = q.shape[1]
    H_kv = k.shape[1]
    if H != H_kv:
        gqa_ratio = H // H_kv
        k = k.repeat_interleave(gqa_ratio, dim=1)
        v = v.repeat_interleave(gqa_ratio, dim=1)

    # [batch=1, H, S, D] layout for F.sdpa
    q_t = q.unsqueeze(0).permute(0, 2, 1, 3)
    k_t = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_t = v.unsqueeze(0).permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale, is_causal=False)
    return out.squeeze(0).permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------
# Helper: one ring step of online softmax (for cp_ring_unfused)
# ---------------------------------------------------------------------------


def _online_softmax_step(acc, e_max, e_sum, q_local, k_cur, v_cur, scale,
                          rank, S_local, kv_rank, H_per_kv, device, H=None):
    """
    One ring step: attend q_local against k_cur/v_cur using SDPA (O(1) memory).

    Uses F.scaled_dot_product_attention (flash-attention backend) to avoid
    materialising the O(S_local^2) score matrix which OOMs at large seq lengths.
    Merges this step's output into the running online-softmax accumulator.

    Returns (acc, e_max, e_sum) — new state after this ring step.
    """
    causal_block = (kv_rank == rank)
    q_off = rank * S_local
    kv_off = kv_rank * S_local
    if H is None:
        H = q_local.shape[1]

    k_exp = k_cur.half().repeat_interleave(H_per_kv, dim=1)
    v_exp = v_cur.half().repeat_interleave(H_per_kv, dim=1)

    q_t = q_local.half().unsqueeze(0).permute(0, 2, 1, 3)   # [1, H, S_q, D]
    k_t = k_exp.unsqueeze(0).permute(0, 2, 1, 3)             # [1, H, S_kv, D]
    v_t = v_exp.unsqueeze(0).permute(0, 2, 1, 3)

    # Build causal mask: q positions [q_off..q_off+S_local], k positions [kv_off..kv_off+S_local]
    q_idx = torch.arange(q_off, q_off + S_local, device=device)
    k_idx = torch.arange(kv_off, kv_off + S_local, device=device)
    allow = q_idx[:, None] >= k_idx[None, :]  # [S_q, S_kv]
    attn_mask = torch.zeros(1, 1, S_local, S_local, device=device, dtype=torch.float16)
    attn_mask = attn_mask.masked_fill(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))

    step_out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=attn_mask, scale=scale
    ).squeeze(0).permute(1, 0, 2).float()  # [S_q, H, D]

    # Compute row_max and p_sum in Q-tiles to avoid O(S_q * S_kv) allocation.
    # Each tile: [1, H, TILE_Q, S_kv] — bounded by TILE_Q * H * S_kv * 4 bytes.
    TILE_Q = 256  # 256 * 32 * 32768 * 4 = 1 GB per tile — acceptable
    row_max = torch.full((S_local, H, 1), float("-inf"), device=device, dtype=torch.float32)
    p_sum   = torch.zeros(S_local, H, 1, device=device, dtype=torch.float32)

    for qi in range(0, S_local, TILE_Q):
        qi_end = min(qi + TILE_Q, S_local)
        q_tile = q_t[:, :, qi:qi_end, :]  # [1, H, tile, D]
        qk_tile = (q_tile.float() * scale) @ k_t.float().transpose(-2, -1)  # [1, H, tile, S_kv]
        allow_tile = allow[qi:qi_end]  # [tile, S_kv]
        qk_tile = qk_tile.masked_fill(~allow_tile.unsqueeze(0).unsqueeze(0), float("-inf"))
        rm = qk_tile.max(dim=-1, keepdim=True).values.squeeze(0).permute(1, 0, 2)  # [tile,H,1]
        ps = torch.exp(qk_tile - rm.unsqueeze(0).permute(0, 2, 1, 3)).sum(
            dim=-1, keepdim=True).squeeze(0).permute(1, 0, 2)  # [tile, H, 1]
        row_max[qi:qi_end] = rm
        p_sum[qi:qi_end] = ps

    n_e_max = torch.maximum(e_max, row_max)
    alpha = torch.exp(e_max - n_e_max)
    acc = acc * alpha + step_out * torch.exp(row_max - n_e_max)
    e_sum = e_sum * alpha + p_sum * torch.exp(row_max - n_e_max)
    e_max = n_e_max
    return acc, e_max, e_sum


# ---------------------------------------------------------------------------
# Primary ablation benchmark
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("impl", [
    "local_flash",
    "tp",
    "cp_allgather_nccl",
    "cp_allgather_iris",
    "cp_ring_unfused",
    "cp_ring_fused",
])
@bench.axis("model", ["llama3-8b", "llama3-70b"])
@bench.axis("seq_len_total", [8192, 32768, 65536, 131072, 256000])
def flash_prefill_ablation(state, ctx):
    """
    Multi-implementation ablation for distributed prefill attention.

    FLOPS formula (uniform across impls for fair scaling efficiency comparison):
        4 * S_total^2 * H * D

    This equals ideal single-GPU equivalent work, so TFLOPS directly measures
    distributed scaling efficiency.
    """
    seq_len_total = state["seq_len_total"]
    model = state["model"]
    impl = state["impl"]

    H, H_kv, D = MODEL_SHAPES[model]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16
    scale = D ** -0.5

    # --- skip rules ---
    if impl == "tp" and H // world_size < 1:
        state.skip("not enough heads to split across ranks")

    if impl.startswith("cp_") and seq_len_total % world_size != 0:
        state.skip(f"seq_len_total={seq_len_total} not divisible by world_size={world_size}")

    # cp_ring_fused is now implemented (Option B, Phase 12):
    # iris ring transfer kernel (same as cp_ring_unfused) +
    # ring_prefill_attn_step_kernel at full grid=(cdiv(S_local,BLOCK_Q), H)
    # instead of Python einsum. No compute/comm overlap yet (that's Option A),
    # but attention runs fully parallel — eliminates the grid=(1,H) bottleneck.

    S_local = seq_len_total // world_size if seq_len_total % world_size == 0 else seq_len_total

    # Tuning: reduce iterations for large sequences
    if seq_len_total >= 131072:
        state.set_warmup(5)
        state.set_repeat(20)
    elif seq_len_total >= 32768:
        state.set_warmup(10)
        state.set_repeat(50)

    # --- FLOPS (uniform formula) ---
    state.set_flops(4 * seq_len_total * seq_len_total * H * D)

    # --- Communication volume ---
    # Both allgather and ring transfer the same total bytes (ring = sequential,
    # allgather = simultaneous; volume is the same)
    elem_size = torch.empty(0, dtype=dtype).element_size()
    kv_bytes_per_rank = S_local * H_kv * D * elem_size
    comm_bytes = 2 * (world_size - 1) * kv_bytes_per_rank  # K + V

    if impl in ("local_flash", "tp"):
        state.add_counter("comm_gb", 0.0)
    else:
        state.add_counter("comm_gb", comm_bytes / 1e9)

    # =========================================================================
    # impl: local_flash — compute ceiling, no comms
    # =========================================================================
    if impl == "local_flash":
        q = torch.randn(S_local, H, D, dtype=dtype, device=device) / math.sqrt(D)
        k = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)
        v = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)

        def run():
            return _sdpa_attn(q, k, v, scale)

        state.exec(run)
        return

    # =========================================================================
    # impl: tp — head-parallel TP, no attention comms
    # =========================================================================
    if impl == "tp":
        H_local = H // world_size
        H_kv_local = max(1, H_kv // world_size)
        q = torch.randn(seq_len_total, H_local, D, dtype=dtype, device=device) / math.sqrt(D)
        k = torch.randn(seq_len_total, H_kv_local, D, dtype=dtype, device=device) / math.sqrt(D)
        v = torch.randn(seq_len_total, H_kv_local, D, dtype=dtype, device=device) / math.sqrt(D)

        def run():
            return _sdpa_attn(q, k, v, scale)

        state.exec(run)
        return

    # =========================================================================
    # impl: cp_allgather_nccl — dist.all_gather KV then local flash attn
    # =========================================================================
    if impl == "cp_allgather_nccl":
        q_local = torch.randn(S_local, H, D, dtype=dtype, device=device) / math.sqrt(D)
        k_local = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)
        v_local = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)

        # Buffers for gathered KV
        k_full = torch.empty(seq_len_total, H_kv, D, dtype=dtype, device=device)
        v_full = torch.empty(seq_len_total, H_kv, D, dtype=dtype, device=device)

        k_chunks = list(k_full.view(world_size, S_local, H_kv, D).unbind(0))
        v_chunks = list(v_full.view(world_size, S_local, H_kv, D).unbind(0))

        def run():
            dist.all_gather(k_chunks, k_local)
            dist.all_gather(v_chunks, v_local)
            # Causal: only attend to ranks <= cur_rank
            k_causal = k_full[:(rank + 1) * S_local]
            v_causal = v_full[:(rank + 1) * S_local]
            return _sdpa_attn(q_local, k_causal, v_causal, scale)

        state.exec(run)
        return

    # =========================================================================
    # impl: cp_allgather_iris — iris.load per KV block fused into attn loop
    # =========================================================================
    if impl == "cp_allgather_iris":
        from iris.x.prefill_attn import distributed_prefill_attn_kernel

        # Allocate KV cache on sym heap (paged, PAGE_SIZE=S_local per rank)
        PAGE_SIZE = min(S_local, 64)  # reasonable page size
        num_blocks = math.ceil(S_local / PAGE_SIZE)

        k_cache = ctx.empty((num_blocks, PAGE_SIZE, H_kv, D), dtype=dtype)
        v_cache = ctx.empty((num_blocks, PAGE_SIZE, H_kv, D), dtype=dtype)

        # Fill with random data
        k_cache.copy_(torch.randn_like(k_cache) / math.sqrt(D))
        v_cache.copy_(torch.randn_like(v_cache) / math.sqrt(D))

        q_local = torch.randn(S_local, H, D, dtype=dtype, device=device) / math.sqrt(D)
        out = torch.empty(S_local, H, D, dtype=dtype, device=device)

        context_tensor = ctx.get_device_context()

        # Build identity local block table
        local_block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

        # Build global block table [batch=1, world_size * num_blocks, 2]
        global_block_table = torch.zeros(1, world_size * num_blocks, 2, dtype=torch.int32, device=device)
        for r in range(world_size):
            start = r * num_blocks
            global_block_table[0, start:start + num_blocks, 0] = r
            global_block_table[0, start:start + num_blocks, 1] = torch.arange(num_blocks)

        seq_lens = torch.full((1,), S_local, dtype=torch.int32, device=device)

        H_PER_KV = H // H_kv
        BLOCK_Q = min(64, triton.next_power_of_2(S_local))
        BLOCK_K = PAGE_SIZE
        HEAD_DIM = triton.next_power_of_2(D)

        ctx.barrier()

        def run():
            distributed_prefill_attn_kernel[
                (1, triton.cdiv(S_local, BLOCK_Q), H)
            ](
                q_local, k_cache, v_cache, global_block_table, seq_lens, out,
                context_tensor,
                q_local.stride(0), q_local.stride(1), q_local.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                global_block_table.stride(0), global_block_table.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                S_local, H, H_kv,
                S_local,  # chunk_len
                scale,
                cur_rank=rank, world_size=world_size,
                H_PER_KV=H_PER_KV, PAGE_SIZE=PAGE_SIZE,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, HEAD_DIM=HEAD_DIM,
            )

        state.exec(run)
        return

    # =========================================================================
    # impl: cp_ring_unfused — iris ring transfer + PyTorch online softmax
    # =========================================================================
    if impl == "cp_ring_unfused":
        q_local = torch.randn(S_local, H, D, dtype=dtype, device=device) / math.sqrt(D)
        k_chunk = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)
        v_chunk = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)

        chunk_elems_per_head = S_local * D
        ring_k_bufs = [ctx.empty(chunk_elems_per_head, dtype=dtype) for _ in range(H_kv)]
        ring_v_bufs = [ctx.empty(chunk_elems_per_head, dtype=dtype) for _ in range(H_kv)]
        signal_flags = ctx.zeros((H_kv,), dtype=torch.int32)
        context_tensor = ctx.get_device_context()
        H_per_kv = H // H_kv
        BLOCK = 256

        def preamble():
            signal_flags.zero_()
            for h in range(H_kv):
                ring_k_bufs[h].copy_(k_chunk[:, h, :].contiguous().view(-1))
                ring_v_bufs[h].copy_(v_chunk[:, h, :].contiguous().view(-1))
            ctx.barrier()

        def run():
            acc = torch.zeros(S_local, H, D, dtype=torch.float32, device=device)
            e_max = torch.full((S_local, H, 1), float("-inf"), dtype=torch.float32, device=device)
            e_sum = torch.zeros(S_local, H, 1, dtype=torch.float32, device=device)
            k_cur = k_chunk.clone()
            v_cur = v_chunk.clone()

            for step in range(world_size):
                kv_rank = (rank - step + world_size) % world_size
                if step > 0:
                    next_k = torch.empty_like(k_cur)
                    next_v = torch.empty_like(v_cur)
                    for h in range(H_kv):
                        _iris_ring_transfer_kernel[(1,)](
                            k_cur[:, h, :].contiguous().view(-1),
                            v_cur[:, h, :].contiguous().view(-1),
                            ring_k_bufs[h], ring_v_bufs[h],
                            signal_flags, chunk_elems_per_head,
                            context_tensor, h,
                            cur_rank=rank, world_size=world_size,
                            BLOCK=BLOCK,
                        )
                        torch.cuda.synchronize()
                        next_k[:, h, :] = ring_k_bufs[h].view(S_local, D)
                        next_v[:, h, :] = ring_v_bufs[h].view(S_local, D)
                    k_cur = next_k
                    v_cur = next_v
                    ctx.barrier()  # all ranks finish iris.store before overwriting ring bufs
                    for h in range(H_kv):
                        ring_k_bufs[h].copy_(k_cur[:, h, :].contiguous().view(-1))
                        ring_v_bufs[h].copy_(v_cur[:, h, :].contiguous().view(-1))

                if kv_rank <= rank:
                    acc, e_max, e_sum = _online_softmax_step(
                        acc, e_max, e_sum, q_local, k_cur, v_cur, scale,
                        rank, S_local, kv_rank, H_per_kv, device, H=H,
                    )

            return (acc / e_sum.clamp(min=1e-9)).to(dtype)

        state.exec(run, preamble_fn=preamble)
        return

    # =========================================================================
    # impl: cp_ring_fused — iris ring transfer + Triton flash attention step
    #
    # Phase 12 (Option B): same iris ring transfer as cp_ring_unfused, but the
    # attention step uses ring_prefill_attn_step_kernel at full grid parallelism
    # grid=(cdiv(S_local, BLOCK_Q), H) instead of Python einsum.
    #
    # Key improvement over cp_ring_unfused: fully parallel attention compute
    # (no Python-level loop overhead or einsum allocation). No compute/comm
    # overlap yet — that requires Option A (persistent kernel, Phase 12A).
    # =========================================================================
    if impl == "cp_ring_fused":
        from iris.x.prefill_attn import (
            ring_prefill_attn_step_kernel,
            finalize_prefill_output_kernel,
        )

        q_local = torch.randn(S_local, H, D, dtype=dtype, device=device) / math.sqrt(D)
        k_chunk = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)
        v_chunk = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)

        chunk_elems_per_head = S_local * D
        ring_k_bufs = [ctx.empty(chunk_elems_per_head, dtype=dtype) for _ in range(H_kv)]
        ring_v_bufs = [ctx.empty(chunk_elems_per_head, dtype=dtype) for _ in range(H_kv)]
        signal_flags = ctx.zeros((H_kv,), dtype=torch.int32)
        context_tensor = ctx.get_device_context()
        BLOCK_TRANSFER = 256

        H_PER_KV = H // H_kv
        BLOCK_Q = min(64, triton.next_power_of_2(S_local))
        BLOCK_K = min(64, triton.next_power_of_2(S_local))
        HEAD_DIM = triton.next_power_of_2(D)

        # HBM-backed accumulator tensors for ring_prefill_attn_step_kernel.
        # acc stored as fp16 between steps (matches kernel's store format).
        # e_max/e_sum stored as fp32 with shape [S_local, H] (2D, no trailing dim).
        k_recv = torch.empty(S_local, H_kv, D, dtype=dtype, device=device)
        v_recv = torch.empty(S_local, H_kv, D, dtype=dtype, device=device)
        acc_buf = torch.empty(S_local, H, HEAD_DIM, dtype=dtype, device=device)
        e_max_buf = torch.empty(S_local, H, dtype=torch.float32, device=device)
        e_sum_buf = torch.empty(S_local, H, dtype=torch.float32, device=device)
        out = torch.empty(S_local, H, HEAD_DIM, dtype=dtype, device=device)

        attn_grid = (triton.cdiv(S_local, BLOCK_Q), H)

        def preamble():
            signal_flags.zero_()
            for h in range(H_kv):
                ring_k_bufs[h].copy_(k_chunk[:, h, :].contiguous().view(-1))
                ring_v_bufs[h].copy_(v_chunk[:, h, :].contiguous().view(-1))
            acc_buf.zero_()
            e_max_buf.fill_(float("-inf"))
            e_sum_buf.zero_()
            ctx.barrier()

        def run():
            k_cur = k_chunk.clone()
            v_cur = v_chunk.clone()

            for step in range(world_size):
                kv_rank = (rank - step + world_size) % world_size

                if step > 0:
                    # --- iris ring transfer (same as cp_ring_unfused) ---
                    next_k = torch.empty_like(k_cur)
                    next_v = torch.empty_like(v_cur)
                    for h in range(H_kv):
                        _iris_ring_transfer_kernel[(1,)](
                            k_cur[:, h, :].contiguous().view(-1),
                            v_cur[:, h, :].contiguous().view(-1),
                            ring_k_bufs[h], ring_v_bufs[h],
                            signal_flags, chunk_elems_per_head,
                            context_tensor, h,
                            cur_rank=rank, world_size=world_size,
                            BLOCK=BLOCK_TRANSFER,
                        )
                        torch.cuda.synchronize()
                        next_k[:, h, :] = ring_k_bufs[h].view(S_local, D)
                        next_v[:, h, :] = ring_v_bufs[h].view(S_local, D)
                    k_cur = next_k
                    v_cur = next_v
                    ctx.barrier()
                    for h in range(H_kv):
                        ring_k_bufs[h].copy_(k_cur[:, h, :].contiguous().view(-1))
                        ring_v_bufs[h].copy_(v_cur[:, h, :].contiguous().view(-1))

                # Stage received KV into contiguous buffer for the Triton kernel
                k_recv.copy_(k_cur)
                v_recv.copy_(v_cur)

                # --- Triton flash attention step at full grid parallelism ---
                # ring_prefill_attn_step_kernel early-returns if kv_rank > cur_rank
                kv_global_off = kv_rank * S_local
                q_global_off = rank * S_local
                ring_prefill_attn_step_kernel[attn_grid](
                    q_ptr=q_local,
                    k_recv_buf_ptr=k_recv,
                    v_recv_buf_ptr=v_recv,
                    acc_ptr=acc_buf,
                    e_max_ptr=e_max_buf,
                    e_sum_ptr=e_sum_buf,
                    stride_qs=q_local.stride(0),
                    stride_qh=q_local.stride(1),
                    stride_qd=q_local.stride(2),
                    stride_kbuf_t=k_recv.stride(0),
                    stride_kbuf_h=k_recv.stride(1),
                    stride_kbuf_d=k_recv.stride(2),
                    stride_acc_t=acc_buf.stride(0),
                    stride_acc_h=acc_buf.stride(1),
                    stride_acc_d=acc_buf.stride(2),
                    stride_e_t=e_max_buf.stride(0),
                    stride_e_h=e_max_buf.stride(1),
                    S_local=S_local,
                    H=H,
                    H_kv=H_kv,
                    chunk_len=S_local,
                    q_global_offset=q_global_off,
                    kv_global_offset=kv_global_off,
                    scale=scale,
                    kv_rank=kv_rank,
                    cur_rank=rank,
                    H_PER_KV=H_PER_KV,
                    BLOCK_Q=BLOCK_Q,
                    BLOCK_K=BLOCK_K,
                    HEAD_DIM=HEAD_DIM,
                )

            # Normalize and write final output
            finalize_prefill_output_kernel[attn_grid](
                acc_ptr=acc_buf,
                e_max_ptr=e_max_buf,
                e_sum_ptr=e_sum_buf,
                out_ptr=out,
                stride_acc_t=acc_buf.stride(0),
                stride_acc_h=acc_buf.stride(1),
                stride_acc_d=acc_buf.stride(2),
                stride_e_t=e_max_buf.stride(0),
                stride_e_h=e_max_buf.stride(1),
                stride_os=out.stride(0),
                stride_oh=out.stride(1),
                stride_od=out.stride(2),
                S_local=S_local,
                H=H,
                BLOCK_Q=BLOCK_Q,
                HEAD_DIM=HEAD_DIM,
            )

        state.exec(run, preamble_fn=preamble)
        return


# ---------------------------------------------------------------------------
# Standalone: ring transfer bandwidth isolation
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("model", ["llama3-8b", "llama3-70b"])
@bench.axis("seq_len_total", [8192, 32768, 65536, 131072, 256000])
def comm_only_ring(state, ctx):
    """
    Isolates ring transfer bandwidth — no attention compute.

    Runs world_size-1 ring steps and measures bytes/sec vs MI300X xGMI theoretical peak.
    Reports BW in GB/s via state.set_bytes().
    """
    seq_len_total = state["seq_len_total"]
    model = state["model"]

    H, H_kv, D = MODEL_SHAPES[model]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    if seq_len_total % world_size != 0:
        state.skip(f"seq_len_total={seq_len_total} not divisible by world_size={world_size}")

    S_local = seq_len_total // world_size

    if seq_len_total >= 131072:
        state.set_warmup(5)
        state.set_repeat(20)
    elif seq_len_total >= 32768:
        state.set_warmup(10)
        state.set_repeat(50)

    chunk_elems_per_head = S_local * D
    ring_k_bufs = [ctx.empty(chunk_elems_per_head, dtype=dtype) for _ in range(H_kv)]
    ring_v_bufs = [ctx.empty(chunk_elems_per_head, dtype=dtype) for _ in range(H_kv)]
    signal_flags = ctx.zeros((H_kv,), dtype=torch.int32)
    context_tensor = ctx.get_device_context()
    BLOCK = 256

    k_local = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)
    v_local = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)

    # Total bytes moved = (world_size - 1) ring steps × 2 (K+V) × S_local × H_kv × D × elem_size
    elem_size = torch.empty(0, dtype=dtype).element_size()
    comm_bytes = (world_size - 1) * 2 * S_local * H_kv * D * elem_size
    state.set_bytes(comm_bytes)

    def preamble():
        signal_flags.zero_()
        for h in range(H_kv):
            ring_k_bufs[h].copy_(k_local[:, h, :].contiguous().view(-1))
            ring_v_bufs[h].copy_(v_local[:, h, :].contiguous().view(-1))
        ctx.barrier()

    def run():
        k_cur = k_local.clone()
        v_cur = v_local.clone()
        for _step in range(world_size - 1):
            next_k = torch.empty_like(k_cur)
            next_v = torch.empty_like(v_cur)
            for h in range(H_kv):
                _iris_ring_transfer_kernel[(1,)](
                    k_cur[:, h, :].contiguous().view(-1),
                    v_cur[:, h, :].contiguous().view(-1),
                    ring_k_bufs[h], ring_v_bufs[h],
                    signal_flags, chunk_elems_per_head,
                    context_tensor, h,
                    cur_rank=rank, world_size=world_size,
                    BLOCK=BLOCK,
                )
                torch.cuda.synchronize()
                next_k[:, h, :] = ring_k_bufs[h].view(S_local, D)
                next_v[:, h, :] = ring_v_bufs[h].view(S_local, D)
            k_cur = next_k
            v_cur = next_v
            for h in range(H_kv):
                ring_k_bufs[h].copy_(k_cur[:, h, :].contiguous().view(-1))
                ring_v_bufs[h].copy_(v_cur[:, h, :].contiguous().view(-1))

    state.exec(run, preamble_fn=preamble)


# ---------------------------------------------------------------------------
# Standalone: NCCL AllGather bandwidth isolation
# ---------------------------------------------------------------------------


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("model", ["llama3-8b", "llama3-70b"])
@bench.axis("seq_len_total", [8192, 32768, 65536, 131072, 256000])
def comm_only_allgather(state, ctx):
    """
    Isolates NCCL AllGather bandwidth — no attention compute.

    AllGathers both K and V tensors. Enables computation of:
      - Compute/comms overlap efficiency: (comm + compute - total) / comm
      - xGMI BW utilization: comm_gb / latency_s vs 336 GB/s theoretical
    """
    seq_len_total = state["seq_len_total"]
    model = state["model"]

    H, H_kv, D = MODEL_SHAPES[model]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    if seq_len_total % world_size != 0:
        state.skip(f"seq_len_total={seq_len_total} not divisible by world_size={world_size}")

    S_local = seq_len_total // world_size

    if seq_len_total >= 131072:
        state.set_warmup(5)
        state.set_repeat(20)
    elif seq_len_total >= 32768:
        state.set_warmup(10)
        state.set_repeat(50)

    k_local = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)
    v_local = torch.randn(S_local, H_kv, D, dtype=dtype, device=device) / math.sqrt(D)

    k_full = torch.empty(seq_len_total, H_kv, D, dtype=dtype, device=device)
    v_full = torch.empty(seq_len_total, H_kv, D, dtype=dtype, device=device)

    k_chunks = list(k_full.view(world_size, S_local, H_kv, D).unbind(0))
    v_chunks = list(v_full.view(world_size, S_local, H_kv, D).unbind(0))

    # AllGather volume: (world_size - 1) / world_size * total_kv_bytes per tensor
    # Standard formula: send (ws-1)/ws fraction, but simpler = ws-1 times S_local
    comm_bytes = 2 * (world_size - 1) * S_local * H_kv * D * 2  # K + V, fp16
    state.set_bytes(comm_bytes)

    def run():
        dist.all_gather(k_chunks, k_local)
        dist.all_gather(v_chunks, v_local)

    state.exec(run)


if __name__ == "__main__":
    bench.main()
