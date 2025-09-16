#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
A simple, minimal example demonstrating how to use the flash_decode_fused_layer.

This script initializes the necessary distributed components with Iris,
creates sample input tensors, instantiates the layer, and calls its
forward pass once. It then prints the shape and a slice of the output
tensor to show that the operation completed successfully.

The layer is defined in the flash_decode_fused_layer.py file.
All the triton kernels are defined in decode_kernels.py
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import iris
import argparse
from flash_decode_fused_layer import flash_decode_fused_layer


def parse_args():
    """Parses command-line arguments for the example."""
    parser = argparse.ArgumentParser(description="A minimal example for flash_decode_fused_layer.")
    parser.add_argument("--kv_len_per_rank", type=int, default=32768, help="KV sequence length per rank.")
    parser.add_argument("--num_heads", type=int, default=96, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=128, help="Dimension of each attention head.")
    parser.add_argument("--num_seqs", type=int, default=4, help="Number of sequences in the batch.")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of GPUs to run the example on.")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="PyTorch data type to use."
    )
    return parser.parse_args()


def setup_example_data(rank, world_size, args, dtype):
    """Creates a set of random tensors to serve as inputs for the layer."""

    num_query_heads = args.num_heads
    # Assume an 8:1 Grouped-Query Attention ratio for this example
    num_kv_heads = max(1, args.num_heads // 8)
    block_size = 1  # PagedAttention works with blocks of tokens

    # Number of blocks needed on this rank to store the KV cache for all sequences
    num_blocks_per_rank = (args.kv_len_per_rank + block_size - 1) // block_size

    print(f"[Rank {rank}] Creating example tensors...")

    # 1. Query tensor: The new tokens for which we are calculating attention.
    query = torch.randn(args.num_seqs, num_query_heads, args.head_dim, dtype=dtype).cuda()

    # 2. Key/Value Caches: Tensors representing the keys and values
    #    The KV is split across ranks
    key_cache_this_rank = torch.randn(num_blocks_per_rank, block_size, num_kv_heads, args.head_dim, dtype=dtype).cuda()
    value_cache_this_rank = torch.randn(
        num_blocks_per_rank, block_size, num_kv_heads, args.head_dim, dtype=dtype
    ).cuda()

    # 3. Block Tables: A mapping that tells the kernel where to find the blocks for each sequence in the KV cache.
    #    Here, we create a simple identity mapping for demonstration.
    block_tables_this_rank = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(args.num_seqs, 1).cuda()

    # 4. Global KV Lengths Tensor: The layer needs to know the sequence length on all ranks.
    # Create a list of lengths for each sequence in the batch on this rank.
    kv_lens_per_rank = [args.kv_len_per_rank] * args.num_seqs
    # Create a 1D tensor from this list. Shape: (NUM_SEQS,)
    kv_lens_tensor_this_rank = torch.tensor(kv_lens_per_rank, dtype=torch.int32).cuda()
    # Reshape to (1, NUM_SEQS) and repeat for all ranks to get shape (world_size, NUM_SEQS)
    global_kv_lens_tensor = kv_lens_tensor_this_rank.unsqueeze(0).repeat(world_size, 1)

    return {
        "query": query,
        "key_cache_this_rank": key_cache_this_rank,
        "value_cache_this_rank": value_cache_this_rank,
        "block_tables_this_rank": block_tables_this_rank,
        "global_kv_lens_tensor": global_kv_lens_tensor,
    }


def example_run(rank: int, world_size: int, init_url: str, args: dict):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=init_url, world_size=world_size, rank=rank)

    # 1. Initialize Iris for distributed communication
    shmem = iris.iris()

    torch.manual_seed(42)
    torch.set_default_device("cuda")
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        print("--- flash_decode_fused_layer Minimal Example ---")
        print(f"Running with {world_size} rank(s).")

    # 2. Set up the example input tensors
    tensor_data = setup_example_data(rank, world_size, args, dtype)
    shmem.barrier()

    # 3. Define the layer's parameters
    num_kv_heads = max(1, args.num_heads // 8)
    scale = args.head_dim**-0.5
    common_params = {
        "num_q_heads": args.num_heads,
        "num_kv_heads": num_kv_heads,
        "q_head_dim": args.head_dim,
        "v_head_dim": args.head_dim,
        "page_size": 1,
        "scale": scale,
        "soft_cap": 0.0,
        "max_allowed_batch": args.num_seqs,
    }

    # 4. Instantiate the layer
    if rank == 0:
        print("\nInstantiating flash_decode_fused_layer...")
    fd_layer = flash_decode_fused_layer(shmem, rank, rank, world_size, world_size, **common_params)

    # 5. Call the forward pass of the layer
    if rank == 0:
        print("Calling the forward pass...")
    output = fd_layer(
        tensor_data["query"],
        tensor_data["key_cache_this_rank"],
        tensor_data["value_cache_this_rank"],
        tensor_data["global_kv_lens_tensor"],
        tensor_data["block_tables_this_rank"],
    )

    # Ensure the computation is finished before printing
    torch.cuda.synchronize()
    shmem.barrier()

    # 6. Print a summary of the output tensor on the main rank
    if rank == 0:
        print("\n--- Example Run Finished ---")
        print(f"Output tensor shape: {output.shape}")
        print("Output tensor values (first 5 elements of the first sequence):")
        print(output[0, 0, :5])
        print("--------------------------")

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args.num_ranks
    init_url = "tcp://127.0.0.1:29500"
    mp.spawn(
        fn=example_run,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
