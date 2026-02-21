# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.all_reduce

Reduces tensors across all ranks using a sum reduction.
Each rank contributes its local tensor; the result on every rank is the element-wise sum.

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

    M, N = 1024, 512
    dtype = torch.float16

    # Each rank fills its tensor with (rank + 1): rank 0 -> 1.0, rank 1 -> 2.0, ...
    input_tensor = shmem.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))

    output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    shmem.ccl.all_reduce(output_tensor, input_tensor)
    torch.cuda.synchronize()

    # Expected: sum of rank+1 for all ranks = 1 + 2 + ... + world_size
    expected = float(world_size * (world_size + 1) // 2)
    assert torch.allclose(output_tensor, torch.full_like(output_tensor, expected), atol=0.5), (
        f"Rank {rank}: all_reduce mismatch. Got {output_tensor[0, 0].item():.1f}, expected {expected:.1f}"
    )

    if rank == 0:
        print(f"iris.ccl.all_reduce: {world_size} ranks, shape ({M}, {N}), dtype {dtype}")
        print(f"  Each rank contributes rank+1; expected sum = {expected:.0f}")
        print(f"  output[0, 0] = {output_tensor[0, 0].item():.1f} ✓")

    shmem.barrier()
    del shmem
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
