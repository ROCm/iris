#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import time

import iris.experimental.iris_rdma as iris_rdma


@triton.jit
def consumer_get_kernel(
    local_ptr,
    remote_ptr,
    n_elements,
    src_rank: tl.constexpr,
    device_ctx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Consumer kernel that enqueues RDMA get operations to pull data.
    Uses symmetric heap model: remote_ptr points to same offset in remote heap.
    After RDMA get completes, data will be available at local_ptr.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Local and remote pointers (same offset in symmetric heap)
    local_ptrs = local_ptr + offsets
    remote_ptrs = remote_ptr + offsets
    
    # Enqueue RDMA GET operation: pull from remote to local
    iris_rdma.get(local_ptrs, remote_ptrs, src_rank, device_ctx, mask)


@triton.jit
def verify_kernel(
    input_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Verification kernel that checks received data.
    Expected pattern: ascending numbers 0, 1, 2, ..., n_elements-1
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load received data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Check if it matches expected pattern (0, 1, 2, 3, ...)
    expected = offsets.to(data.dtype)
    is_correct = (data == expected).to(tl.int32)
    
    tl.store(result_ptr + offsets, is_correct, mask=mask)


def main():
    
    dtype = torch.bfloat16
    
    # Initialize distributed
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device_id = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group(
        backend='nccl',
        device_id=device_id
    )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size < 2:
        print("This example requires at least 2 ranks")
        sys.exit(1)
    
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    
    print(f"[Rank {rank}/{world_size}] Initialized on {device}")
    
    # Create Iris RDMA context with queue
    heap_size = 1024 * 1024 * 8  # 8MB
    queue_size = 512
    ctx = iris_rdma.iris(heap_size=heap_size, queue_size=queue_size)
    
    print(f"[Rank {rank}] Iris RDMA initialized")
    print(f"[Rank {rank}]   - Heap base: {ctx.get_heap_base():#x}")
    print(f"[Rank {rank}]   - Queue ptr: {ctx.get_queue_ptr():#x}")
    
    # Get device context for Triton kernels
    device_ctx = ctx.get_device_context()
    
    # Allocate buffers in symmetric heap
    n_elements = 4091
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Allocate on the symmetric heap
    local_buffer = ctx.zeros(n_elements, dtype=dtype)
    
    ctx.barrier()
    
    # ============================================================
    # SERVER (Rank 1): Prepare data for RDMA get
    # ============================================================
    if rank == 1:
        print(f"\n[Rank 1] === Server: Preparing Data ===")
        
        # Fill buffer with data using PyTorch
        print(f"[Rank 1] Filling buffer with data...")
        local_buffer.copy_(torch.arange(n_elements, dtype=dtype, device=device))
        torch.cuda.synchronize()
        print(f"[Rank 1] Data ready, first 10: {local_buffer[:10].tolist()}")
        print(f"[Rank 1] Waiting for client to pull data...")
    
    # ============================================================
    # CLIENT (Rank 0): Pull data using RDMA get
    # ============================================================
    if rank == 0:
        print(f"\n[Rank 0] === Client: Pulling Data via RDMA GET ===")
        src_rank = 1
        
        # Launch RDMA GET enqueue kernel
        print(f"[Rank 0] Launching RDMA GET kernel to pull from Rank {src_rank}...")
        consumer_get_kernel[grid](
            local_buffer,      # local destination
            local_buffer,      # remote source (same offset in symmetric heap)
            n_elements,
            src_rank=src_rank,
            device_ctx=device_ctx,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Wait for GPU to finish enqueueing
        torch.cuda.synchronize()
        print(f"[Rank 0] RDMA GET operations enqueued to queue")
    
    ctx.barrier()
    print(f"[Rank {rank}] Barrier complete, all RDMA operations finished")
    
    # ============================================================
    # CLIENT (Rank 0): Verify pulled data
    # ============================================================
    if rank == 0:
        print(f"\n[Rank 0] === Verifying Pulled Data ===")
        
        # Show received data
        print(f"[Rank 0] Received data first 10: {local_buffer[:10].tolist()}")
        
        # Verify data (use int32 for result buffer - stores 0 or 1 for correctness)
        result_buffer = torch.zeros(n_elements, dtype=torch.int32, device=device)
        
        verify_kernel[grid](
            local_buffer,
            result_buffer,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        result_cpu = result_buffer.cpu()
        num_correct = result_cpu.sum().item()
        num_total = n_elements
        
        print(f"[Rank 0] Verified: {int(num_correct)}/{num_total}")
        
        if num_correct == num_total:
            print(f"\n" + "="*60)
            print(f"[Rank 0] SUCCESS! Data pulled correctly via RDMA GET!")
        else:
            print(f"[Rank 0] FAILED - Data mismatch!")
            first_wrong_idx = (result_cpu == 0).nonzero(as_tuple=True)[0]
            if len(first_wrong_idx) > 0:
                idx = first_wrong_idx[0].item()
                print(f"[Rank 0]   First wrong at index {idx}")
                print(f"[Rank 0]   Expected: {idx}")
                print(f"[Rank 0]   Got: {local_buffer[idx].item()}")
            sys.exit(1)
    
    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

