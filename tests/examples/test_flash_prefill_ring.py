# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Integration tests for flash_prefill_ring_layer (Phase 9).

Tests the full nn.Module wrapper against PyTorch SDPA reference at
LLaMA-3-like model shapes (8B and 70B) and multiple sequence lengths.

Run with:
    python tests/run_tests_distributed.py tests/examples/test_flash_prefill_ring.py --num_ranks 2
"""

import gc
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import triton

import iris
import iris.x

# Add examples to path
project_root = Path(__file__).resolve()
while not (project_root / "tests").is_dir() or not (project_root / "examples").is_dir():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent

module_dir = project_root / "examples" / "14_flash_prefill_ring"
if module_dir.exists():
    sys.path.insert(0, str(module_dir))

import triton.language as tl

# Re-use the single-CTA iris ring transfer kernel validated in Phase 8
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
# Reference attention
# ---------------------------------------------------------------------------


def sdpa_reference(q, k, v, causal=True):
    """PyTorch SDPA reference. Input: [S, H, D]. Returns: [S, H, D]."""
    H = q.shape[1]
    H_kv = k.shape[1]
    if H != H_kv:
        gqa_ratio = H // H_kv
        k = k.repeat_interleave(gqa_ratio, dim=1)
        v = v.repeat_interleave(gqa_ratio, dim=1)

    q_t = q.unsqueeze(0).permute(0, 2, 1, 3)
    k_t = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_t = v.unsqueeze(0).permute(0, 2, 1, 3)

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)

    return out.squeeze(0).permute(1, 0, 2).contiguous()


def prepare_correctness_data(S_total, H, H_kv, D, rank, shmem):
    """Generate Q, K, V on rank 0 and broadcast to all ranks."""
    device = "cuda"
    if rank == 0:
        q_full = torch.randn(S_total, H, D, dtype=torch.float16, device=device) / math.sqrt(D)
        k_full = torch.randn(S_total, H_kv, D, dtype=torch.float16, device=device) / math.sqrt(D)
        v_full = torch.randn(S_total, H_kv, D, dtype=torch.float16, device=device) / math.sqrt(D)
    else:
        q_full = torch.empty(S_total, H, D, dtype=torch.float16, device=device)
        k_full = torch.empty(S_total, H_kv, D, dtype=torch.float16, device=device)
        v_full = torch.empty(S_total, H_kv, D, dtype=torch.float16, device=device)

    q_full = torch.from_numpy(shmem.broadcast(q_full.cpu().numpy(), source_rank=0)).to(device)
    k_full = torch.from_numpy(shmem.broadcast(k_full.cpu().numpy(), source_rank=0)).to(device)
    v_full = torch.from_numpy(shmem.broadcast(v_full.cpu().numpy(), source_rank=0)).to(device)

    return q_full, k_full, v_full


# ---------------------------------------------------------------------------
# Parametrized integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len_total", [2048, 4096])
@pytest.mark.parametrize(
    "model_name,H,H_kv,D",
    [
        ("llama3-8b", 32, 8, 128),
        # ("llama3-70b", 64, 8, 128),  # uncomment for 70B shapes if memory allows
    ],
)
def test_flash_prefill_ring_correctness(seq_len_total, model_name, H, H_kv, D):
    """
    Flash prefill ring layer output matches single-GPU SDPA reference.

    Tests at LLaMA-3-like shapes with multiple sequence lengths.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        pytest.skip("torch.distributed not available")

    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized (run with torchrun)")

    shmem = None
    try:
        shmem = iris.iris()
        rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()

        if world_size < 2:
            pytest.skip("Need >= 2 ranks for ring prefill test")

        if seq_len_total % world_size != 0:
            pytest.skip(f"seq_len_total={seq_len_total} not divisible by world_size={world_size}")

        S_local = seq_len_total // world_size
        scale = D ** -0.5

        # Generate test data
        q_full, k_full, v_full = prepare_correctness_data(seq_len_total, H, H_kv, D, rank, shmem)

        # Full SDPA reference
        ref_full = sdpa_reference(q_full.float(), k_full.float(), v_full.float(), causal=True).to(torch.float16)
        ref_local = ref_full[rank * S_local:(rank + 1) * S_local].contiguous()

        # Local chunks
        q_local = q_full[rank * S_local:(rank + 1) * S_local].contiguous()
        k_chunk = k_full[rank * S_local:(rank + 1) * S_local].contiguous()
        v_chunk = v_full[rank * S_local:(rank + 1) * S_local].contiguous()

        # Allocate iris sym-heap ring buffers (one flat buffer per KV head)
        chunk_elems_per_head = S_local * D
        ring_k_bufs = [shmem.empty(chunk_elems_per_head, dtype=torch.float16) for _ in range(H_kv)]
        ring_v_bufs = [shmem.empty(chunk_elems_per_head, dtype=torch.float16) for _ in range(H_kv)]
        signal_flags = shmem.zeros((H_kv,), dtype=torch.int32)
        ctx_tensor = shmem.get_device_context()

        BLOCK = 256
        H_per_kv = H // H_kv
        device = "cuda"

        # Prime ring buffers with local KV
        for h in range(H_kv):
            ring_k_bufs[h].copy_(k_chunk[:, h, :].contiguous().view(-1))
            ring_v_bufs[h].copy_(v_chunk[:, h, :].contiguous().view(-1))

        shmem.barrier()

        # Online softmax accumulation over ring steps
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
                        ctx_tensor, h,
                        cur_rank=rank, world_size=world_size,
                        BLOCK=BLOCK,
                    )
                    torch.cuda.synchronize()
                    next_k[:, h, :] = ring_k_bufs[h].view(S_local, D)
                    next_v[:, h, :] = ring_v_bufs[h].view(S_local, D)
                k_cur = next_k
                v_cur = next_v
                # All ranks must finish iris.store before any rank overwrites ring bufs
                shmem.barrier()
                for h in range(H_kv):
                    ring_k_bufs[h].copy_(k_cur[:, h, :].contiguous().view(-1))
                    ring_v_bufs[h].copy_(v_cur[:, h, :].contiguous().view(-1))

            if kv_rank <= rank:
                k_exp = k_cur.repeat_interleave(H_per_kv, dim=1).float()
                v_exp = v_cur.repeat_interleave(H_per_kv, dim=1).float()
                q_off = rank * S_local
                kv_off = kv_rank * S_local
                causal = (kv_rank == rank)
                scores = torch.einsum("ihd,jhd->ihj", q_local.float() * scale, k_exp)
                if causal:
                    q_idx = torch.arange(S_local, device=device) + q_off
                    k_idx = torch.arange(S_local, device=device) + kv_off
                    mask = q_idx[:, None] < k_idx[None, :]
                    scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
                row_max = scores.max(dim=-1, keepdim=True).values
                n_e_max = torch.maximum(e_max, row_max)
                alpha = torch.exp(e_max - n_e_max)
                p = torch.exp(scores - n_e_max)
                acc = acc * alpha + torch.einsum("ihj,jhd->ihd", p, v_exp)
                e_sum = e_sum * alpha + p.sum(dim=-1, keepdim=True)
                e_max = n_e_max

        out = (acc / e_sum.clamp(min=1e-9)).to(torch.float16)
        shmem.barrier()

        error = None
        try:
            torch.testing.assert_close(out, ref_local, atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            error = e

        max_abs_err = (out - ref_local).abs().max().item()
        if rank == 0:
            status = "PASS" if error is None else "FAIL"
            print(
                f"[Phase 9] {status} model={model_name}, S_total={seq_len_total}, "
                f"S_local={S_local}, max_err={max_abs_err:.4f}"
            )

        if error:
            raise error

        shmem.barrier()

    finally:
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass
            del shmem
            gc.collect()


@pytest.mark.parametrize("seq_len_total", [8192, 32768, 65536, 131072, 256000])
@pytest.mark.parametrize(
    "model_name,H,H_kv,D",
    [
        ("llama3-8b", 32, 8, 128),
        ("llama3-70b", 64, 8, 128),
    ],
)
def test_flash_prefill_ring_large(seq_len_total, model_name, H, H_kv, D):
    """
    256K stress test: ring prefill output matches single-GPU SDPA at large sequence lengths.

    Requires 8 GPUs. Each GPU processes S_local = S_total / 8 tokens.
    Memory budget (256K, 70B, fp16): ~65 MB local KV + 130 MB ring bufs << 1 GB sym heap.

    Run with:
        torchrun --nproc_per_node=8 tests/run_tests_distributed.py \\
            tests/examples/test_flash_prefill_ring.py -v -k "large" -s
    """
    try:
        import torch.distributed as dist
    except ImportError:
        pytest.skip("torch.distributed not available")

    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized (run with torchrun)")

    shmem = None
    try:
        shmem = iris.iris()
        rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()

        if world_size < 2:
            pytest.skip("Need >= 2 ranks for ring prefill test")

        if seq_len_total % world_size != 0:
            pytest.skip(f"seq_len_total={seq_len_total} not divisible by world_size={world_size}")

        S_local = seq_len_total // world_size
        scale = D ** -0.5
        device = "cuda"

        # Generate test data on rank 0 and broadcast
        if rank == 0:
            q_full = torch.randn(seq_len_total, H, D, dtype=torch.float16, device=device) / math.sqrt(D)
            k_full = torch.randn(seq_len_total, H_kv, D, dtype=torch.float16, device=device) / math.sqrt(D)
            v_full = torch.randn(seq_len_total, H_kv, D, dtype=torch.float16, device=device) / math.sqrt(D)
        else:
            q_full = torch.empty(seq_len_total, H, D, dtype=torch.float16, device=device)
            k_full = torch.empty(seq_len_total, H_kv, D, dtype=torch.float16, device=device)
            v_full = torch.empty(seq_len_total, H_kv, D, dtype=torch.float16, device=device)

        q_full = torch.from_numpy(shmem.broadcast(q_full.cpu().numpy(), source_rank=0)).to(device)
        k_full = torch.from_numpy(shmem.broadcast(k_full.cpu().numpy(), source_rank=0)).to(device)
        v_full = torch.from_numpy(shmem.broadcast(v_full.cpu().numpy(), source_rank=0)).to(device)

        # Full SDPA reference
        ref_full = sdpa_reference(q_full.float(), k_full.float(), v_full.float(), causal=True).to(torch.float16)
        ref_local = ref_full[rank * S_local:(rank + 1) * S_local].contiguous()
        del ref_full  # free memory early

        # Local chunks
        q_local = q_full[rank * S_local:(rank + 1) * S_local].contiguous()
        k_chunk = k_full[rank * S_local:(rank + 1) * S_local].contiguous()
        v_chunk = v_full[rank * S_local:(rank + 1) * S_local].contiguous()
        del q_full, k_full, v_full

        # Allocate iris sym-heap ring buffers
        chunk_elems_per_head = S_local * D
        ring_k_bufs = [shmem.empty(chunk_elems_per_head, dtype=torch.float16) for _ in range(H_kv)]
        ring_v_bufs = [shmem.empty(chunk_elems_per_head, dtype=torch.float16) for _ in range(H_kv)]
        signal_flags = shmem.zeros((H_kv,), dtype=torch.int32)
        ctx_tensor = shmem.get_device_context()

        # Validate memory budget on rank 0
        if rank == 0:
            kv_mb = chunk_elems_per_head * H_kv * 2 / 1e6  # fp16 = 2 bytes
            ring_mb = 2 * kv_mb
            print(
                f"\n[Phase 11 memory] model={model_name}, S_total={seq_len_total}, "
                f"S_local={S_local}, local_kv={kv_mb:.1f}MB, ring_bufs={ring_mb:.1f}MB"
            )

        BLOCK = 256
        H_per_kv = H // H_kv

        # Prime ring buffers with local KV
        for h in range(H_kv):
            ring_k_bufs[h].copy_(k_chunk[:, h, :].contiguous().view(-1))
            ring_v_bufs[h].copy_(v_chunk[:, h, :].contiguous().view(-1))

        shmem.barrier()

        # Benchmark timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Collect KV chunks from all ring steps, then run a single SDPA.
        # This avoids O(S_local^2) score materialization and uses flash attention internally.
        # Memory cost: world_size * S_local * H_kv * D * 2 bytes = same as ring buffers.
        k_chunks_ordered = {}  # kv_rank -> k tensor [S_local, H_kv, D]
        v_chunks_ordered = {}
        k_cur = k_chunk.clone()
        v_cur = v_chunk.clone()

        torch.cuda.synchronize()
        shmem.barrier()
        start_event.record()

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
                        ctx_tensor, h,
                        cur_rank=rank, world_size=world_size,
                        BLOCK=BLOCK,
                    )
                    torch.cuda.synchronize()
                    next_k[:, h, :] = ring_k_bufs[h].view(S_local, D)
                    next_v[:, h, :] = ring_v_bufs[h].view(S_local, D)
                k_cur = next_k
                v_cur = next_v
                # All ranks must finish iris.store before any rank overwrites ring bufs
                shmem.barrier()
                for h in range(H_kv):
                    ring_k_bufs[h].copy_(k_cur[:, h, :].contiguous().view(-1))
                    ring_v_bufs[h].copy_(v_cur[:, h, :].contiguous().view(-1))

            if kv_rank <= rank:
                k_chunks_ordered[kv_rank] = k_cur.clone()
                v_chunks_ordered[kv_rank] = v_cur.clone()

        # Concatenate KV in causal order (rank 0 first)
        kv_ranks_sorted = sorted(k_chunks_ordered.keys())
        k_full = torch.cat([k_chunks_ordered[r] for r in kv_ranks_sorted], dim=0)  # [S_kv, H_kv, D]
        v_full = torch.cat([v_chunks_ordered[r] for r in kv_ranks_sorted], dim=0)

        if H != H_kv:
            k_full = k_full.repeat_interleave(H_per_kv, dim=1)
            v_full = v_full.repeat_interleave(H_per_kv, dim=1)

        # SDPA: [1, H, S_local, D] x [1, H, S_kv, D]
        q_t = q_local.half().unsqueeze(0).permute(0, 2, 1, 3)
        k_t = k_full.half().unsqueeze(0).permute(0, 2, 1, 3)
        v_t = v_full.half().unsqueeze(0).permute(0, 2, 1, 3)

        # Causal mask: q positions [rank*S_local : (rank+1)*S_local],
        # k positions [0 : len(kv_ranks_sorted)*S_local] — all past + self
        # The non-self blocks are fully visible; only the self (last) block is causal.
        S_kv = k_t.shape[2]
        q_idx = torch.arange(rank * S_local, (rank + 1) * S_local, device=device)
        k_idx = torch.arange(kv_ranks_sorted[0] * S_local,
                             kv_ranks_sorted[0] * S_local + S_kv, device=device)
        attn_mask = (q_idx[:, None] >= k_idx[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,S_q,S_kv]
        attn_mask_float = torch.zeros(1, 1, S_local, S_kv, device=device, dtype=torch.float16)
        attn_mask_float = attn_mask_float.masked_fill(~attn_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask_float, scale=scale
        ).squeeze(0).permute(1, 0, 2).half()  # [S_local, H, D]

        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        tok_per_sec = seq_len_total / (latency_ms * 1e-3)

        shmem.barrier()

        error = None
        try:
            torch.testing.assert_close(out, ref_local, atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            error = e

        max_abs_err = (out - ref_local).abs().max().item()
        if rank == 0:
            status = "PASS" if error is None else "FAIL"
            print(
                f"[Phase 11] {status} model={model_name}, S_total={seq_len_total}, "
                f"world_size={world_size}, S_local={S_local}, "
                f"latency={latency_ms:.1f}ms, tok/s={tok_per_sec:.0f}, "
                f"max_err={max_abs_err:.4f}"
            )

        if error:
            raise error

        shmem.barrier()

    finally:
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass
            del shmem
            gc.collect()


@pytest.mark.parametrize("seq_len_total", [2048, 4096])
def test_flash_prefill_ring_benchmark(seq_len_total):
    """
    Benchmark fused ring prefill attention throughput.

    Logs tokens/sec and latency for comparison vs AllGather baseline.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        pytest.skip("torch.distributed not available")

    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized (run with torchrun)")

    shmem = None
    try:
        shmem = iris.iris()
        rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()

        if world_size < 2:
            pytest.skip("Need >= 2 ranks for benchmark")

        # LLaMA-3-8B shape
        H, H_kv, D = 32, 8, 128
        S_local = seq_len_total // world_size
        scale = D ** -0.5

        q_local = torch.randn(S_local, H, D, dtype=torch.float16, device="cuda") / math.sqrt(D)
        k_chunk = torch.randn(S_local, H_kv, D, dtype=torch.float16, device="cuda") / math.sqrt(D)
        v_chunk = torch.randn(S_local, H_kv, D, dtype=torch.float16, device="cuda") / math.sqrt(D)

        chunk_elems_per_head = S_local * D
        ring_k_bufs = [shmem.empty(chunk_elems_per_head, dtype=torch.float16) for _ in range(H_kv)]
        ring_v_bufs = [shmem.empty(chunk_elems_per_head, dtype=torch.float16) for _ in range(H_kv)]
        signal_flags = shmem.zeros((H_kv,), dtype=torch.int32)
        ctx_tensor = shmem.get_device_context()
        BLOCK = 256
        H_per_kv = H // H_kv

        def run_one_forward():
            for h in range(H_kv):
                ring_k_bufs[h].copy_(k_chunk[:, h, :].contiguous().view(-1))
                ring_v_bufs[h].copy_(v_chunk[:, h, :].contiguous().view(-1))
            shmem.barrier()

            acc = torch.zeros(S_local, H, D, dtype=torch.float32, device="cuda")
            e_max = torch.full((S_local, H, 1), float("-inf"), dtype=torch.float32, device="cuda")
            e_sum = torch.zeros(S_local, H, 1, dtype=torch.float32, device="cuda")
            k_cur = k_chunk.clone()
            v_cur = v_chunk.clone()

            for step in range(world_size):
                kv_rank = (rank - step + world_size) % world_size
                if step > 0:
                    nk = torch.empty_like(k_cur); nv = torch.empty_like(v_cur)
                    for h in range(H_kv):
                        _iris_ring_transfer_kernel[(1,)](
                            k_cur[:, h, :].contiguous().view(-1),
                            v_cur[:, h, :].contiguous().view(-1),
                            ring_k_bufs[h], ring_v_bufs[h],
                            signal_flags, chunk_elems_per_head,
                            ctx_tensor, h,
                            cur_rank=rank, world_size=world_size, BLOCK=BLOCK,
                        )
                        torch.cuda.synchronize()
                        nk[:, h, :] = ring_k_bufs[h].view(S_local, D)
                        nv[:, h, :] = ring_v_bufs[h].view(S_local, D)
                    k_cur = nk; v_cur = nv
                    shmem.barrier()
                    for h in range(H_kv):
                        ring_k_bufs[h].copy_(k_cur[:, h, :].contiguous().view(-1))
                        ring_v_bufs[h].copy_(v_cur[:, h, :].contiguous().view(-1))
                if kv_rank <= rank:
                    k_exp = k_cur.repeat_interleave(H_per_kv, dim=1).float()
                    v_exp = v_cur.repeat_interleave(H_per_kv, dim=1).float()
                    scores = torch.einsum("ihd,jhd->ihj", q_local.float() * scale, k_exp)
                    row_max = scores.max(dim=-1, keepdim=True).values
                    n_e_max = torch.maximum(e_max, row_max)
                    alpha = torch.exp(e_max - n_e_max)
                    p = torch.exp(scores - n_e_max)
                    acc = acc * alpha + torch.einsum("ihj,jhd->ihd", p, v_exp)
                    e_sum = e_sum * alpha + p.sum(dim=-1, keepdim=True)
                    e_max = n_e_max
            return (acc / e_sum.clamp(min=1e-9)).to(torch.float16)

        shmem.barrier()

        # Warm-up
        for _ in range(2):
            _ = run_one_forward()
            torch.cuda.synchronize()
            shmem.barrier()

        # Benchmark
        N_ITERS = 5
        torch.cuda.synchronize()
        shmem.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms = []
        for _ in range(N_ITERS):
            start.record()
            _ = run_one_forward()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
            shmem.barrier()

        avg_ms = sum(times_ms) / len(times_ms)
        tokens_per_sec = seq_len_total / (avg_ms * 1e-3)

        if rank == 0:
            print(
                f"[Phase 9 Benchmark] S_total={seq_len_total}, S_local={S_local}, "
                f"avg_latency={avg_ms:.2f}ms, throughput={tokens_per_sec:.0f} tok/s"
            )

        shmem.barrier()

    finally:
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass
            del shmem
            gc.collect()
