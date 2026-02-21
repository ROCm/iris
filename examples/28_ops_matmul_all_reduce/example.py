# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.matmul_all_reduce

Fused matrix multiplication and all-reduce.
Computes: output = all_reduce(A @ B) where A and B are replicated across ranks.
Each rank computes the same local GEMM; the results are summed across all ranks.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py
"""

import gc
import os

import torch
import torch.distributed as dist

import iris


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    shmem = iris.iris(heap_size=2**31)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, K, N = 512, 256, 128
    dtype = torch.float16

    # Identical A and B on every rank (replicated weights scenario)
    torch.manual_seed(42)
    A = shmem.randn((M, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    shmem.ops.matmul_all_reduce(output, A, B)
    torch.cuda.synchronize()

    # Reference: local GEMM scaled by world_size (each rank adds the same result)
    A_ref = A.clone().float()
    B_ref = B.clone().float()
    ref = torch.matmul(A_ref, B_ref).to(dtype) * world_size

    assert torch.allclose(output.float(), ref.float(), atol=1.0, rtol=0.05), (
        f"Rank {rank}: matmul_all_reduce mismatch. Max diff: {(output.float() - ref.float()).abs().max().item():.4f}"
    )

    if rank == 0:
        print(f"iris.ops.matmul_all_reduce: {world_size} ranks, A ({M},{K}), B ({K},{N}), dtype {dtype}")
        print(f"  output[0, 0] = {output[0, 0].item():.4f} ✓")

    shmem.barrier()
    del shmem
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
