#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
A simple, minimal example demonstrating how to use the FDFusedLayer.

This script initializes the necessary distributed components with Iris,
creates sample input tensors, instantiates the layer, and calls its
forward pass once. It then prints the shape and a slice of the output
tensor to show that the operation completed successfully.

The layer is defined in the fd_fused_layer.py file.
All the triton kernels are defined in decode_kernels.py
"""

import torch
import iris

# Since this file is at the same level as the layer file, a direct import works.
from fd_fused_layer import FDFusedLayer

# ==============================================================================
# Example Configuration
# ==============================================================================
# These parameters define the shape of the problem for this example.
# They are kept small for a quick and simple run.
KV_LEN_PER_RANK = 32768
NUM_HEADS = 96
HEAD_DIM = 128
NUM_SEQS = 4
DTYPE = torch.float16

# ==============================================================================
# Helper Function to Create Example Data
# ==============================================================================

def setup_example_data(rank, world_size):
    """Creates a set of random tensors to serve as inputs for the layer."""

    num_query_heads = NUM_HEADS
    # Assume an 8:1 Grouped-Query Attention ratio for this example
    num_kv_heads = max(1, NUM_HEADS // 8)
    block_size = 1 # PagedAttention works with blocks of tokens

    # Number of blocks needed on this rank to store the KV cache for all sequences
    num_blocks_per_rank = (KV_LEN_PER_RANK + block_size - 1) // block_size

    print(f"[Rank {rank}] Creating example tensors...")

    # 1. Query tensor: The new tokens for which we are calculating attention.
    query = torch.randn(NUM_SEQS, num_query_heads, HEAD_DIM, dtype=DTYPE).cuda()

    # 2. Key/Value Caches: Tensors representing the keys and values
    #    The KV is split across ranks
    key_cache_this_rank = torch.randn(num_blocks_per_rank, block_size, num_kv_heads, HEAD_DIM, dtype=DTYPE).cuda()
    value_cache_this_rank = torch.randn(num_blocks_per_rank, block_size, num_kv_heads, HEAD_DIM, dtype=DTYPE).cuda()

    # 3. Block Tables: A mapping that tells the kernel where to find the blocks for each sequence in the KV cache.
    #    Here, we create a simple identity mapping for demonstration.
    block_tables_this_rank = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(NUM_SEQS, 1).cuda()

    # 4. Global KV Lengths Tensor: The layer needs to know the sequence length on all ranks.
    # Create a list of lengths for each sequence in the batch on this rank.
    kv_lens_per_rank = [KV_LEN_PER_RANK] * NUM_SEQS
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


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # 1. Initialize Iris for distributed communication
    _iris = iris.iris()
    rank = _iris.get_rank()
    world_size = _iris.get_num_ranks()

    torch.manual_seed(42)
    torch.set_default_device("cuda")

    if rank == 0:
        print("--- FDFusedLayer Minimal Example ---")
        print(f"Running with {world_size} rank(s).")

    # 2. Set up the example input tensors
    tensor_data = setup_example_data(rank, world_size)
    _iris.barrier()

    # 3. Define the layer's parameters
    num_kv_heads = max(1, NUM_HEADS // 8)
    scale = HEAD_DIM**-0.5
    common_params = {
        "num_q_heads": NUM_HEADS,
        "num_kv_heads": num_kv_heads,
        "q_head_dim": HEAD_DIM,
        "v_head_dim": HEAD_DIM,
        "page_size": 1,
        "scale": scale,
        "soft_cap": 0.0,
        "max_allowed_batch": NUM_SEQS,
    }

    # 4. Instantiate the layer
    if rank == 0:
        print("\nInstantiating FDFusedLayer...")
    fd_layer = FDFusedLayer(_iris, rank, rank, world_size, world_size, **common_params)

    # 5. Call the forward pass of the layer
    if rank == 0:
        print("Calling the forward pass...")
    output = fd_layer(
        tensor_data['query'],
        tensor_data['key_cache_this_rank'],
        tensor_data['value_cache_this_rank'],
        tensor_data['global_kv_lens_tensor'],
        tensor_data['block_tables_this_rank']
    )

    # Ensure the computation is finished before printing
    torch.cuda.synchronize()
    _iris.barrier()

    # 6. Print a summary of the output tensor on the main rank
    if rank == 0:
        print("\n--- Example Run Finished ---")
        print(f"Output tensor shape: {output.shape}")
        print("Output tensor values (first 5 elements of the first sequence):")
        print(output[0, 0, :5])
        print("--------------------------")

    _iris.barrier()
