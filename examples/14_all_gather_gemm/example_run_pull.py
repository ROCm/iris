#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
A simple, minimal example demonstrating how to use the persistent_ag_gemm
'Pull' model kernel for a distributed All-Gather + GEMM operation.

This script initializes Iris and Torch Distributed, creates sample input
tensors, launches the fused Triton kernel, and validates the
output against a standard PyTorch reference implementation.

The kernel is defined in the all_gather_gemm_pull.py file.
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import iris
import argparse
from all_gather_gemm_pull import persistent_ag_gemm


def parse_args():
    """Parses command-line arguments for the example."""
    parser = argparse.ArgumentParser(description="A minimal example for a fused All-Gather + GEMM (Pull Model).")
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

    # The total K dimension is sharded across all ranks.
    K_total = args.K
    if K_total % world_size != 0:
        raise ValueError("K dimension must be divisible by the world size for this example.")
    K_local = K_total // world_size

    # Create the full A and B matrices on rank 0
    if rank == 0:
        A_global = torch.randn(args.M, K_total, dtype=dtype, device="cuda")
        B_global = torch.randn(K_total, args.N, dtype=dtype, device="cuda")
    else:
        A_global = torch.empty(args.M, K_total, dtype=dtype, device="cuda")
        B_global = torch.empty(K_total, args.N, dtype=dtype, device="cuda")

    # Broadcast the full matrices to all ranks to ensure data consistency
    dist.broadcast(A_global, src=0)
    dist.broadcast(B_global, src=0)

    # Each rank takes its local, vertical slice of A
    A_local_shard = A_global[:, rank * K_local : (rank + 1) * K_local].contiguous()

    return {
        "A_local_shard": A_local_shard,
        "B_global": B_global,  # B remains replicated
        "A_global_for_validation": A_global,  # Keep the full A for the reference calculation
    }


def example_run(rank: int, world_size: int, init_url: str, args: argparse.Namespace):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend, init_method=init_url, world_size=world_size, rank=rank, device_id=torch.device(f"cuda:{rank}")
    )

    # Initialize Iris for distributed communication
    shmem = iris.iris()

    torch.manual_seed(42)  # Use a fixed seed for consistent random data
    torch.cuda.set_device(rank)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        print("--- Fused All-Gather + GEMM (Pull Model) Minimal Example ---")
        print(f"Running with {world_size} rank(s).")

    # Set up the example input tensors
    tensor_data = setup_example_data(rank, world_size, args, dtype)
    shmem.barrier()

    # Prepare for the kernel launch
    A_original = tensor_data["A_local_shard"]
    B = tensor_data["B_global"]

    # Allocate a tensor in Iris's shared memory heap for remote access
    A_iris = shmem.empty(A_original.shape, dtype=A_original.dtype)
    A_iris.copy_(A_original)

    C_fused = torch.empty(args.M, args.N, dtype=dtype).cuda()  # Output tensor for our kernel

    NUM_SMS = torch.cuda.get_device_properties(rank).multi_processor_count
    grid = (NUM_SMS,)

    # Launch the fused Triton kernel
    if rank == 0:
        print("\nLaunching persistent_ag_gemm kernel...")
    persistent_ag_gemm[grid](
        A_iris,
        B,
        C_fused,
        args.M,
        args.N,
        args.K,
        A_iris.stride(0),
        A_iris.stride(1),
        B.stride(0),
        B.stride(1),
        C_fused.stride(0),
        C_fused.stride(1),
        BLOCK_SIZE_M=args.BLOCK_SIZE_M,
        BLOCK_SIZE_N=args.BLOCK_SIZE_N,
        BLOCK_SIZE_K=args.BLOCK_SIZE_K,
        GROUP_SIZE_M=args.GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        NUM_XCDS=1,
        EVEN_K=(args.K % args.BLOCK_SIZE_K == 0),
        heap_bases=shmem.get_heap_bases(),
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
        # Calculate the reference solution using torch.matmul
        C_ref = torch.matmul(tensor_data["A_global_for_validation"], B)

        # Compare the results
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
