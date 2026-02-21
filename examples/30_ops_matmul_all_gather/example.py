# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.matmul_all_gather

Fused matrix multiplication and all-gather along the M dimension.
Computes: output = all_gather(A @ B) where A is row-sharded across ranks.
Each rank computes its local GEMM; results are gathered so every rank holds the full (M, N) output.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py
"""

import os

import torch
import torch.distributed as dist

import iris


def run(shmem):
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M = 512  # Full M dimension (must be divisible by world_size)
    K = 256
    N = 128
    dtype = torch.float16

    if M % world_size != 0:
        if rank == 0:
            print(f"Skipping: M={M} not divisible by world_size={world_size}")
        return

    M_local = M // world_size  # Each rank handles M_local rows

    # Each rank has its own row shard of A
    torch.manual_seed(42 + rank)
    A_local = shmem.randn((M_local, K), dtype=dtype)

    # B is replicated on all ranks
    torch.manual_seed(0)
    B = shmem.randn((K, N), dtype=dtype)

    # Output is the full (M, N) matrix on every rank
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    shmem.ops.matmul_all_gather(output, A_local, B)
    torch.cuda.synchronize()

    # Reference: compute local GEMM and all-gather along M
    C_local = torch.matmul(A_local.float(), B.clone().float()).to(dtype)
    C_shards = [torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(C_shards, C_local)
    ref = torch.cat(C_shards, dim=0)  # (M, N)

    assert torch.allclose(output.float(), ref.float(), atol=1.0, rtol=0.05), (
        f"Rank {rank}: matmul_all_gather mismatch. Max diff: {(output.float() - ref.float()).abs().max().item():.4f}"
    )

    if rank == 0:
        print(f"iris.ops.matmul_all_gather: {world_size} ranks, A_local ({M_local},{K}), B ({K},{N}), dtype {dtype}")
        print(f"  output shape: {tuple(output.shape)}, output[0, 0] = {output[0, 0].item():.4f} ✓")


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    shmem = iris.iris(heap_size=2**31)
    try:
        run(shmem)
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
