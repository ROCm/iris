#!/usr/bin/env python3
"""RS with raw tl.load on pre-translated peer pointers — bypass iris.load entirely.

The symmetric heap maps peer memory into our VA space. We can compute
peer_ptr = heap_bases[peer] + (local_ptr - heap_bases[local])
and use tl.load directly. No iris.load address translation per element.
"""

import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)


@triton.jit
def raw_load_reduce_scatter_kernel(
    input_ptr, output_ptr,
    M, N, M_local,
    stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """RS using raw tl.load on symmetric heap pointers — no iris.load."""
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_offset = cur_rank * num_m_tiles

    # Pre-compute base translation: peer_base - local_base (constant per rank)
    local_base = tl.load(heap_bases + cur_rank).to(tl.uint64)

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        local_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

        if is_full:
            # Accumulate from all peers using raw tl.load on translated pointers
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for peer in tl.static_range(world_size):
                peer_base = tl.load(heap_bases + peer).to(tl.uint64)
                # Translate: peer_ptr = peer_base + (input_ptr - local_base) + element_offset
                # input_ptr is in the symmetric heap, so (input_ptr - local_base) is the heap offset
                input_as_uint64 = input_ptr.to(tl.uint64)
                peer_input_base = (peer_base + input_as_uint64 - local_base).to(input_ptr.type)
                peer_ptrs = peer_input_base + local_offset
                tile = tl.load(peer_ptrs, cache_modifier=".cv")
                acc += tile.to(acc_dtype)

            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            for peer in tl.static_range(world_size):
                peer_base = tl.load(heap_bases + peer).to(tl.uint64)
                input_as_uint64 = input_ptr.to(tl.uint64)
                peer_input_base = (peer_base + input_as_uint64 - local_base).to(input_ptr.type)
                peer_ptrs = peer_input_base + local_offset
                tile = tl.load(peer_ptrs, mask=mask, other=0.0, cache_modifier=".cv")
                acc += tile.to(acc_dtype)

            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty), mask=out_mask)


# Setup
M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

input_tensor = shmem.zeros((M, N), dtype=dtype)
input_tensor.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
heap_bases = shmem.get_heap_bases()
shmem.barrier()

if rank == 0:
    print(f"Raw tl.load RS: M={M}, N={N}, TP={world_size}")

# RCCL baseline
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
input_rccl = input_tensor.clone()
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters
if rank == 0:
    print(f"RCCL RS: {rccl_ms:.3f}ms")

# Sweep
configs = [(64, 64), (128, 64), (128, 128), (256, 64), (256, 128)]
sms_list = [16, 32, 64, 128]
best_ms = 999.0
best_cfg = None

for bm, bn in configs:
    if M_local % bm != 0:
        continue
    for num_sms in sms_list:
        out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
        shmem.barrier()

        for _ in range(warmup):
            raw_load_reduce_scatter_kernel[(num_sms,)](
                input_tensor, out, M, N, M_local,
                input_tensor.stride(0), input_tensor.stride(1),
                out.stride(0), out.stride(1),
                heap_bases, rank, world_size, bm, bn, num_sms,
            )
        torch.cuda.synchronize()

        # Correctness
        ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
        dist.reduce_scatter_tensor(ref, input_tensor.clone(), op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        max_diff = torch.abs(out - ref).max().item()

        if max_diff > 1.0:
            if rank == 0:
                print(f"  bm={bm:3d} bn={bn:3d} sms={num_sms:3d}: FAIL (diff={max_diff:.2f})")
            continue

        s.record()
        for _ in range(iters):
            raw_load_reduce_scatter_kernel[(num_sms,)](
                input_tensor, out, M, N, M_local,
                input_tensor.stride(0), input_tensor.stride(1),
                out.stride(0), out.stride(1),
                heap_bases, rank, world_size, bm, bn, num_sms,
            )
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        bw = M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9
        if rank == 0:
            print(f"  bm={bm:3d} bn={bn:3d} sms={num_sms:3d}: {ms:.3f}ms ({bw:.1f} GB/s)")
        if ms < best_ms:
            best_ms = ms
            best_cfg = (bm, bn, num_sms)

if rank == 0:
    print()
    print(f"RCCL RS:      {rccl_ms:.3f}ms")
    print(f"Raw load RS:  {best_ms:.3f}ms (bm={best_cfg[0]} bn={best_cfg[1]} sms={best_cfg[2]})")
    print(f"Speedup:      {rccl_ms / best_ms:.2f}x")
    print(f"vs iris.load: {'FASTER' if best_ms < 0.099 else 'same or slower'} (iris.load best: 0.099ms)")

shmem.barrier()
dist.destroy_process_group()
