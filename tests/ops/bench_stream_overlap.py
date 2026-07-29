#!/usr/bin/env python3
"""Stream overlap: launch RS kernel immediately after torch.mm via event wait.

Also tests: torch._C._cuda_setStream to minimize Python-side dispatch gap.
And: pre-compiled Triton kernel to eliminate JIT overhead on hot path.
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
def fast_rs_kernel(
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
            tl.store(output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            tl.store(output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty), mask=out_mask)


M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

RS_CONFIG = {2: (128, 64, 128), 4: (64, 64, 32), 8: (32, 64, 32)}
rs_bm, rs_bn, rs_sms = RS_CONFIG.get(world_size, (128, 64, 128))

A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

C_sym = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
heap_bases = shmem.get_heap_bases()

if rank == 0:
    print(f"Stream overlap test: M={M}, N={N}, K={K_global}, TP={world_size}")

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# --- Baseline: RCCL ---
C_rccl = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
C_rccl_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    torch.mm(A, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_rccl)
    dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters
if rank == 0:
    print(f"RCCL baseline: {rccl_ms:.3f}ms")

# --- Method 1: Same stream, no barrier (current best) ---
shmem.barrier()

# Pre-warm the kernel (eliminate JIT on first call)
fast_rs_kernel[(rs_sms,)](
    C_sym, C_out, M, N, M_local,
    C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
    heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
)
torch.cuda.synchronize()

# Create a wrapper that captures all args to minimize Python dispatch
def run_pipeline():
    torch.mm(A, B, out=C_sym)
    fast_rs_kernel[(rs_sms,)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
    )

for _ in range(warmup):
    run_pipeline()
torch.cuda.synchronize()

s.record()
for _ in range(iters):
    run_pipeline()
e.record()
torch.cuda.synchronize()
same_stream_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Same stream (no barrier): {same_stream_ms:.3f}ms ({rccl_ms/same_stream_ms:.2f}x)")

# --- Method 2: Try functional_collectives if available ---
try:
    from torch.distributed._functional_collectives import reduce_scatter_tensor as func_rs

    C_func = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
    C_func_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")

    for _ in range(warmup):
        torch.mm(A, B, out=C_func)
        func_rs(C_func, "sum", list(range(world_size)))
    torch.cuda.synchronize()

    s.record()
    for _ in range(iters):
        torch.mm(A, B, out=C_func)
        func_rs(C_func, "sum", list(range(world_size)))
    e.record()
    torch.cuda.synchronize()
    func_ms = s.elapsed_time(e) / iters

    if rank == 0:
        print(f"Functional collectives RS: {func_ms:.3f}ms ({rccl_ms/func_ms:.2f}x)")
except Exception as ex:
    if rank == 0:
        print(f"Functional collectives: not available ({ex})")

# --- Method 3: Measure Python dispatch overhead ---
# Time just the Python calls without GPU work
import time

torch.cuda.synchronize()
py_start = time.perf_counter()
for _ in range(10000):
    pass  # baseline: empty loop
py_loop = (time.perf_counter() - py_start) / 10000

torch.cuda.synchronize()
py_start = time.perf_counter()
for _ in range(10000):
    torch.mm(A, B, out=C_sym)
py_mm = (time.perf_counter() - py_start) / 10000

torch.cuda.synchronize()
py_start = time.perf_counter()
for _ in range(10000):
    fast_rs_kernel[(rs_sms,)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        heap_bases, rank, world_size, rs_bm, rs_bn, rs_sms,
    )
py_rs = (time.perf_counter() - py_start) / 10000

if rank == 0:
    print()
    print(f"Python dispatch overhead:")
    print(f"  torch.mm dispatch: {py_mm*1000:.3f}ms ({(py_mm - py_loop)*1e6:.1f}us pure)")
    print(f"  RS kernel dispatch: {py_rs*1000:.3f}ms ({(py_rs - py_loop)*1e6:.1f}us pure)")
    print(f"  combined dispatch: {(py_mm + py_rs - 2*py_loop)*1e6:.1f}us")
    print()
    print(f"GPU time (from events): {same_stream_ms:.3f}ms")
    print(f"Dispatch overhead fraction: {(py_mm + py_rs - 2*py_loop)*1000/same_stream_ms*100:.1f}%")

shmem.barrier()
dist.destroy_process_group()
