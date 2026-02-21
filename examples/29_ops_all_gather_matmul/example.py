# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ops.all_gather_matmul

Fused all-gather and matrix multiplication.
Computes: output = all_gather(A_sharded) @ B
where A is column-sharded across ranks (each rank holds A[:, k_start:k_end]).

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

    M = 512
    K_local = 128  # Each rank holds K_local columns of A
    K = K_local * world_size  # Full K dimension after gather
    N = 256

    dtype = torch.float16

    # Each rank has a different shard of A
    torch.manual_seed(42 + rank)
    A_sharded = shmem.randn((M, K_local), dtype=dtype)

    # B is replicated on all ranks
    torch.manual_seed(0)
    B = shmem.randn((K, N), dtype=dtype)

    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    shmem.ops.all_gather_matmul(output, A_sharded, B)
    torch.cuda.synchronize()

    # Reference: gather A shards from all ranks, then matmul
    A_shards = [torch.zeros(M, K_local, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(A_shards, A_sharded)
    A_full = torch.cat(A_shards, dim=1)  # (M, K)
    ref = torch.matmul(A_full.float(), B.clone().float()).to(dtype)

    assert torch.allclose(output.float(), ref.float(), atol=1.0, rtol=0.05), (
        f"Rank {rank}: all_gather_matmul mismatch. Max diff: {(output.float() - ref.float()).abs().max().item():.4f}"
    )

    if rank == 0:
        print(f"iris.ops.all_gather_matmul: {world_size} ranks, A_sharded ({M},{K_local}), B ({K},{N}), dtype {dtype}")
        print(f"  output shape: {tuple(output.shape)}, output[0, 0] = {output[0, 0].item():.4f} ✓")

    shmem.barrier()
    del shmem
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
