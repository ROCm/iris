#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
A simple, minimal example demonstrating how to use the 'Push' model kernels
for a distributed All-Gather + GEMM operation.

This script initializes Iris and Torch Distributed, creates sample input
tensors, and then launches the two-kernel pipeline:
1. A 'push' kernel broadcasts local shards of matrix 'A' to all GPUs.
2. A 'wait-and-compute' kernel waits for data and performs the GEMM.
Finally, it validates the output against a PyTorch reference.

The kernels are defined in the all_gather_gemm_push.py file.
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import iris
import argparse

# Assume the kernels are in a file named all_gather_gemm_push.py
from all_gather_gemm_push import push_shards_kernel, gemm_push_kernel


def parse_args():
    """Parses command-line arguments for the example."""
    parser = argparse.ArgumentParser(description="A minimal example for a fused All-Gather + GEMM (Push Model).")
    parser.add_argument("--M", type=int, default=128, help="M dimension of the GEMM.")
    parser.add_argument("--N", type=int, default=256, help="N dimension of the GEMM.")
    parser.add_argument("--K", type=int, default=8192, help="Total K dimension of the GEMM (will be sharded).")
    parser.add_argument("--BLOCK_SIZE_M", type=int, default=256, help="Triton kernel tile size for M dimension.")
    parser.add_argument("--BLOCK_SIZE_N", type=int, default=64, help="Triton kernel tile size for N dimension.")
    parser.add_argument("--BLOCK_SIZE_K", type=int, default=64, help="Triton kernel tile size for K dimension.")
    parser.add_argument("--GROUP_SIZE_M", type=int, default=6, help="Triton kernel group size for M dimension.")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of GPUs to run the example on.")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="PyTorch data type to use."
    )
    return parser.parse_args()


def setup_example_data(rank, world_size, args, dtype):
    """Creates a set of random tensors to serve as inputs for the kernel."""
    print(f"[Rank {rank}] Creating example tensors...")

    K_total = args.K
    if K_total % world_size != 0:
        raise ValueError("K dimension must be divisible by the world size for this example.")
    K_local = K_total // world_size

    if rank == 0:
        A_global = torch.randn(args.M, K_total, dtype=dtype, device="cuda")
        B_global = torch.randn(K_total, args.N, dtype=dtype, device="cuda")
    else:
        A_global = torch.empty(args.M, K_total, dtype=dtype, device="cuda")
        B_global = torch.empty(K_total, args.N, dtype=dtype, device="cuda")

    dist.broadcast(A_global, src=0)
    dist.broadcast(B_global, src=0)

    A_local_shard = A_global[:, rank * K_local : (rank + 1) * K_local].contiguous()

    return {
        "A_local_shard": A_local_shard,
        "B_global": B_global,
        "A_global_for_validation": A_global,
    }


def example_run(rank: int, world_size: int, init_url: str, args: argparse.Namespace):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=init_url, world_size=world_size, rank=rank)

    shmem = iris.iris()
    torch.manual_seed(42)
    torch.cuda.set_device(rank)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        print("--- Fused All-Gather + GEMM (Push Model) Minimal Example ---")
        print(f"Running with {world_size} rank(s).")

    tensor_data = setup_example_data(rank, world_size, args, dtype)
    shmem.barrier()

    # Prepare for the kernel launch
    A_original = tensor_data["A_local_shard"]
    B = tensor_data["B_global"]
    C_fused = torch.empty(args.M, args.N, dtype=dtype).cuda()

    # Allocate tensors in Iris shared memory
    A_local_iris = shmem.empty(A_original.shape, dtype=A_original.dtype)
    A_local_iris.copy_(A_original)

    # Create an "inbox" on each rank to receive data from others
    A_inbox_iris = shmem.empty((world_size, args.M, A_original.shape[1]), dtype=A_original.dtype)

    # Create flags for synchronization
    num_m_tiles = (args.M + args.BLOCK_SIZE_M - 1) // args.BLOCK_SIZE_M
    num_k_tiles = (A_original.shape[1] + args.BLOCK_SIZE_K - 1) // args.BLOCK_SIZE_K
    signal_flags = shmem.zeros((world_size, world_size, num_m_tiles, num_k_tiles), dtype=torch.int32)

    NUM_SMS = torch.cuda.get_device_properties(rank).multi_processor_count

    # Launch the two-kernel Push pipeline
    if rank == 0:
        print("\nLaunching push_shards_kernel and gemm_push_kernel...")

    # Define grid for the push kernel
    push_grid = (num_m_tiles, num_k_tiles)
    push_shards_kernel[push_grid](
        A_local_iris,
        A_inbox_iris,
        signal_flags,
        args.M,
        A_local_iris.shape[1],
        A_local_iris.stride(0),
        A_local_iris.stride(1),
        A_inbox_iris.stride(0),
        A_inbox_iris.stride(1),
        A_inbox_iris.stride(2),
        signal_flags.stride(0),
        signal_flags.stride(1),
        signal_flags.stride(2),
        signal_flags.stride(3),
        BLOCK_SIZE_M=args.BLOCK_SIZE_M,
        BLOCK_SIZE_K=args.BLOCK_SIZE_K,
        cur_rank=rank,
        world_size=world_size,
        heap_bases=shmem.get_heap_bases(),
    )

    # Define grid for the GEMM kernel
    gemm_grid = (NUM_SMS,)
    gemm_push_kernel[gemm_grid](
        A_inbox_iris,
        B,
        C_fused,
        args.M,
        args.N,
        args.K,
        signal_flags,
        A_inbox_iris.stride(0),
        A_inbox_iris.stride(1),
        A_inbox_iris.stride(2),
        B.stride(0),
        B.stride(1),
        C_fused.stride(0),
        C_fused.stride(1),
        signal_flags.stride(0),
        signal_flags.stride(1),
        signal_flags.stride(2),
        signal_flags.stride(3),
        BLOCK_SIZE_M=args.BLOCK_SIZE_M,
        BLOCK_SIZE_N=args.BLOCK_SIZE_N,
        BLOCK_SIZE_K=args.BLOCK_SIZE_K,
        GROUP_SIZE_M=args.GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        NUM_XCDS=1,
        EVEN_K=(A_local_iris.shape[1] % args.BLOCK_SIZE_K == 0),
        cur_rank=rank,
        world_size=world_size,
    )

    torch.cuda.synchronize()
    shmem.barrier()
    dist.barrier()

    # Print a summary and perform validation
    if rank == 0:
        print("\n--- Example Run Finished ---")
        print(f"Output tensor C shape: {C_fused.shape}")
        print("Output tensor C values (first 5 elements of the first row):")
        print(C_fused[0, :5])
        print("--------------------------")

        print("\n--- Validation ---")
        C_ref = torch.matmul(tensor_data["A_global_for_validation"], B)
        try:
            torch.testing.assert_close(C_fused, C_ref, atol=1.0, rtol=0.1)
            print("✅ Validation PASSED")
        except AssertionError as e:
            print("❌ Validation FAILED")
            print(e)
        print("------------------")

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args.num_ranks
    init_url = "tcp://127.0.0.1:29504"
    mp.spawn(
        fn=example_run,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
