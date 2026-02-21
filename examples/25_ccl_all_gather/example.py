# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.all_gather

Gathers tensors from all ranks and concatenates them along dimension 0.
Each rank contributes an (M, N) tensor; the result on every rank is (world_size * M, N).

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

    M, N = 512, 256
    dtype = torch.float16

    # Each rank fills its tensor with (rank + 1)
    input_tensor = shmem.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))

    # Output gathers from all ranks: shape (world_size * M, N)
    output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    shmem.barrier()
    shmem.ccl.all_gather(output_tensor, input_tensor)
    torch.cuda.synchronize()

    # Expected: output[r*M:(r+1)*M] == (r + 1) for each rank r
    for r in range(world_size):
        expected = float(r + 1)
        chunk = output_tensor[r * M : (r + 1) * M]
        assert torch.allclose(chunk, torch.full_like(chunk, expected), atol=0.5), (
            f"Rank {rank}: all_gather chunk {r} mismatch. Got {chunk[0, 0].item():.1f}, expected {expected:.1f}"
        )

    if rank == 0:
        print(f"iris.ccl.all_gather: {world_size} ranks, input shape ({M}, {N}), dtype {dtype}")
        print(f"  Output shape: {tuple(output_tensor.shape)}")
        print(f"  output[0, 0] = {output_tensor[0, 0].item():.1f} (rank 0 value = 1.0) ✓")


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
