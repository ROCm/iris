# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.all_to_all

Each rank sends a distinct slice of its input to every other rank.
Input and output are both (M, N * world_size): input[:, r*N:(r+1)*N] is sent to rank r.

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

    M, N = 512, 128
    dtype = torch.float16

    # Build input: the slice destined for rank r is filled with float(rank * 10 + r + 1)
    input_tensor = shmem.zeros((M, N * world_size), dtype=dtype)
    for target_rank in range(world_size):
        value = float(rank * 10 + target_rank + 1)
        input_tensor[:, target_rank * N : (target_rank + 1) * N] = value

    output_tensor = shmem.zeros((M, N * world_size), dtype=dtype)

    shmem.barrier()
    shmem.ccl.all_to_all(output_tensor, input_tensor)
    torch.cuda.synchronize()

    # Expected: output slice from source rank s is float(s * 10 + rank + 1)
    for src_rank in range(world_size):
        expected = float(src_rank * 10 + rank + 1)
        chunk = output_tensor[:, src_rank * N : (src_rank + 1) * N]
        assert torch.allclose(chunk, torch.full_like(chunk, expected), atol=0.5), (
            f"Rank {rank}: all_to_all chunk from rank {src_rank} mismatch. "
            f"Got {chunk[0, 0].item():.1f}, expected {expected:.1f}"
        )

    if rank == 0:
        print(f"iris.ccl.all_to_all: {world_size} ranks, input shape ({M}, {N * world_size}), dtype {dtype}")
        print(f"  output[0, 0] = {output_tensor[0, 0].item():.1f} (from rank 0 -> rank 0 slice = 1.0) ✓")


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
