#!/usr/bin/env python3
"""1D flat RS kernel — contiguous chunk reads instead of 2D tiled reads.

Hypothesis: 2D tiling hurts XGMI burst performance. Flat sequential reads
should achieve higher bandwidth by maximizing cache line utilization.
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
def flat_reduce_scatter_kernel(
    input_ptr, output_ptr,
    total_elements,
    offset_start,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """1D flat RS: each WG reads contiguous chunks from all peers."""
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    for chunk_start in range(pid * BLOCK_SIZE, total_elements, NUM_SMS * BLOCK_SIZE):
        offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        # Global offset in the full M×N buffer
        global_offsets = offset_start + offsets

        base_ptr = input_ptr + global_offsets

        # One-shot pull from all peers
        start_rank = pid % world_size
        acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask).to(acc_dtype)
        for i in tl.static_range(1, world_size):
            r = (start_rank + i) % world_size
            acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask).to(acc_dtype)

        tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

input_tensor = shmem.zeros((M, N), dtype=dtype)
input_tensor.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
output_tensor = torch.zeros(M_local * N, dtype=dtype, device=f"cuda:{rank}")

heap_bases = shmem.get_heap_bases()
shmem.barrier()

total_elements = M_local * N
offset_start = rank * M_local * N  # flat offset into M×N buffer

if rank == 0:
    print(f"Flat RS: M={M}, N={N}, TP={world_size}, total_elements={total_elements}")

# RCCL baseline
input_rccl = input_tensor.clone()
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"RCCL RS: {rccl_ms:.3f}ms")

# Sweep block sizes and SMS
configs = [
    (1024, 32), (1024, 64), (1024, 128),
    (2048, 32), (2048, 64), (2048, 128),
    (4096, 32), (4096, 64), (4096, 128),
    (8192, 32), (8192, 64),
]

best_ms = 999.0
best_cfg = None

for block_size, num_sms in configs:
    out = torch.zeros(M_local * N, dtype=dtype, device=f"cuda:{rank}")

    shmem.barrier()
    for _ in range(warmup):
        flat_reduce_scatter_kernel[(num_sms,)](
            input_tensor, out,
            total_elements, offset_start,
            heap_bases, rank, world_size,
            block_size, num_sms,
        )
    torch.cuda.synchronize()

    # Correctness
    ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    dist.reduce_scatter_tensor(ref, input_tensor.clone(), op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    max_diff = torch.abs(out.view(M_local, N) - ref).max().item()

    if max_diff > 1.0:
        if rank == 0:
            print(f"  bs={block_size:5d} sms={num_sms:3d}: FAIL (diff={max_diff:.2f})")
        continue

    s.record()
    for _ in range(iters):
        flat_reduce_scatter_kernel[(num_sms,)](
            input_tensor, out,
            total_elements, offset_start,
            heap_bases, rank, world_size,
            block_size, num_sms,
        )
    e.record()
    torch.cuda.synchronize()

    ms = s.elapsed_time(e) / iters
    bw = M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9
    if rank == 0:
        print(f"  bs={block_size:5d} sms={num_sms:3d}: {ms:.3f}ms ({bw:.1f} GB/s) diff={max_diff:.4f}")
    if ms < best_ms:
        best_ms = ms
        best_cfg = (block_size, num_sms)

if rank == 0:
    print()
    print(f"RCCL RS:    {rccl_ms:.3f}ms")
    print(f"Flat RS:    {best_ms:.3f}ms (bs={best_cfg[0]}, sms={best_cfg[1]})")
    print(f"Speedup:    {rccl_ms / best_ms:.2f}x")
    print(f"vs 2D tiled RS: {'faster' if best_ms < 0.099 else 'same or slower'} (2D best: 0.099ms)")

shmem.barrier()
dist.destroy_process_group()
