#!/usr/bin/env python3
"""Debug script for GEMM+RS HBM buffer kernel — isolate GEMM correctness."""

import os
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

import iris
from iris.ops import FusedConfig
from iris.ops.matmul_reduce_scatter_hbm_buffer import (
    matmul_reduce_scatter_hbm_buffer,
    matmul_reduce_scatter_hbm_buffer_preamble,
)

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)

M, N, K_local = 1024, 128, 64
M_local = M // world_size
dtype = torch.float16

# Create inputs — same A and B on all ranks for debugging
torch.manual_seed(42)
A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

# Reference: local GEMM partial sum
ref_partial = torch.mm(A, B)
# Reference: reduce-scatter
ref_local = torch.empty((M_local, N), dtype=dtype, device=f"cuda:{rank}")
dist.reduce_scatter_tensor(ref_local, ref_partial, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

# Iris kernel
iris_A = shmem.zeros((M, K_local), dtype=dtype)
iris_A.copy_(A)
iris_B = B.clone()
iris_C = torch.zeros((M_local, N), dtype=dtype, device=f"cuda:{rank}")

config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)
shmem.barrier()

ws = matmul_reduce_scatter_hbm_buffer_preamble(shmem, iris_A, iris_B, config)
matmul_reduce_scatter_hbm_buffer(
    shmem, iris_C, iris_A, iris_B, config=config, workspace=ws, num_scatter_sms=32,
)
torch.cuda.synchronize()

# Check GEMM phase: staged_c should contain A @ B (partial sum)
staged_c = ws.aux_buffer
staged_diff = torch.abs(staged_c - ref_partial).max().item()
print(f"Rank {rank}: staged_c vs torch.mm max_diff = {staged_diff:.6f}")

# Check first tile of staged_c
print(f"Rank {rank}: staged_c[0:4, 0:4] =\n{staged_c[0:4, 0:4]}")
print(f"Rank {rank}: ref_partial[0:4, 0:4] =\n{ref_partial[0:4, 0:4]}")

# Check output
out_diff = torch.abs(iris_C - ref_local).max().item()
print(f"Rank {rank}: output vs reference max_diff = {out_diff:.6f}")

# Check first tile of output
m_start = rank * M_local
print(f"Rank {rank}: iris_C[0:4, 0:4] =\n{iris_C[0:4, 0:4]}")
print(f"Rank {rank}: ref_local[0:4, 0:4] =\n{ref_local[0:4, 0:4]}")

# Check flags
flags = ws.locks
print(f"Rank {rank}: flags sum = {flags.sum().item()}, expected = {flags.numel()}")
print(f"Rank {rank}: flags[:10] = {flags[:10].tolist()}")

shmem.barrier()
dist.destroy_process_group()
