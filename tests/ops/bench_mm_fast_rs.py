#!/usr/bin/env python3
"""End-to-end benchmark: torch.mm + fast iris RS vs torch.mm + RCCL RS."""

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
def fast_reduce_scatter_kernel(
    input_ptr, output_ptr,
    M, N, M_local,
    stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_offset = cur_rank * num_m_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + in_offset
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

        if is_full:
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty), mask=out_mask)


M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 50, 200
NUM_SMS = 304
BM, BN = 256, 64

A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

# Symmetric heap buffer for GEMM output (input to RS)
C_partial = shmem.zeros((M, N), dtype=dtype)
C_output = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

heap_bases = shmem.get_heap_bases()

if rank == 0:
    print(f"E2E benchmark: M={M}, N={N}, K={K_global}, TP={world_size}, dtype={dtype}")
    print()

# --- torch.mm + RCCL RS ---
C_rccl = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_rccl_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")

for _ in range(warmup):
    torch.mm(A, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(iters):
    torch.mm(A, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
end_evt.record()
torch.cuda.synchronize()
rccl_ms = start_evt.elapsed_time(end_evt) / iters

if rank == 0:
    print(f"torch.mm + RCCL RS: {rccl_ms:.3f}ms")

# --- torch.mm + fast iris RS ---
shmem.barrier()

for _ in range(warmup):
    torch.mm(A, B, out=C_partial)
    fast_reduce_scatter_kernel[(NUM_SMS,)](
        C_partial, C_output,
        M, N, M_local,
        C_partial.stride(0), C_partial.stride(1),
        C_output.stride(0), C_output.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_SMS,
    )
torch.cuda.synchronize()

# Correctness
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_partial)
shmem.barrier()
fast_reduce_scatter_kernel[(NUM_SMS,)](
    C_partial, C_output,
    M, N, M_local,
    C_partial.stride(0), C_partial.stride(1),
    C_output.stride(0), C_output.stride(1),
    heap_bases, rank, world_size,
    BM, BN, NUM_SMS,
)
torch.cuda.synchronize()

C_ref = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_ref)
dist.reduce_scatter_tensor(ref, C_ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

max_diff = torch.abs(C_output - ref).max().item()
if rank == 0:
    print(f"Correctness: max_diff = {max_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")

# Benchmark
shmem.barrier()
start_evt.record()
for _ in range(iters):
    torch.mm(A, B, out=C_partial)
    fast_reduce_scatter_kernel[(NUM_SMS,)](
        C_partial, C_output,
        M, N, M_local,
        C_partial.stride(0), C_partial.stride(1),
        C_output.stride(0), C_output.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_SMS,
    )
end_evt.record()
torch.cuda.synchronize()
iris_ms = start_evt.elapsed_time(end_evt) / iters

if rank == 0:
    print(f"torch.mm + fast iris RS: {iris_ms:.3f}ms")
    print()
    speedup = rccl_ms / iris_ms
    print(f"Speedup: {speedup:.2f}x {'🔥 FASTER' if speedup > 1.0 else 'slower'}")

shmem.barrier()
dist.destroy_process_group()
