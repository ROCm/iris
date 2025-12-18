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
def producer_put_kernel(
    buffer_ptr,
    n_elements,
    dst_rank: tl.constexpr,
    device_ctx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Producer kernel that enqueues RDMA put operations.
    Data must already be in buffer_ptr (filled by fill_data_kernel).
    Uses symmetric heap model: same buffer offset in local and remote heap.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Src and dst are the same pointer
    ptrs = buffer_ptr + offsets
    
    # Enqueue RDMA operation
    iris_rdma.put(ptrs, ptrs, dst_rank, device_ctx, mask)


@triton.jit
def consumer_kernel(
    input_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Consumer kernel that verifies received data.
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
    # PRODUCER (Rank 0): Generate data and RDMA put to Rank 1
    # ============================================================
    if rank == 0:
        print(f"\n[Rank 0] === Producer: Generating and Sending Data ===")
        dst_rank = 1
        
        # Step 1: Fill buffer with data using PyTorch (no race condition)
        print(f"[Rank 0] Filling buffer with data using PyTorch...")
        local_buffer.copy_(torch.arange(n_elements, dtype=dtype, device=device))
        print(f"[Rank 0] Data filled, first 10: {local_buffer[:10].tolist()}")
        
        # Step 2: Launch RDMA enqueue kernel (data already in memory)
        print(f"[Rank 0] Launching RDMA enqueue kernel...")
        producer_put_kernel[grid](
            local_buffer,
            n_elements,
            dst_rank=dst_rank,
            device_ctx=device_ctx,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Wait for GPU to finish enqueueing
        torch.cuda.synchronize()
        print(f"[Rank 0] RDMA operations enqueued to queue")
    
    ctx.barrier()
    print(f"[Rank {rank}] Barrier complete, all RDMA operations finished")
    
    # ============================================================
    # CONSUMER (Rank 1): Verify received data
    # ============================================================
    if rank == 1:
        print(f"\n[Rank 1] === Consumer: Verifying Received Data ===")
        
        # Show received data
        print(f"[Rank 1] Received data first 10: {local_buffer[:10].tolist()}")
        
        # Verify data (use int32 for result buffer - stores 0 or 1 for correctness)
        result_buffer = torch.zeros(n_elements, dtype=torch.int32, device=device)
        
        consumer_kernel[grid](
            local_buffer,
            result_buffer,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        result_cpu = result_buffer.cpu()
        num_correct = result_cpu.sum().item()
        num_total = n_elements
        
        print(f"[Rank 1] Verified: {int(num_correct)}/{num_total}")
        
        if num_correct == num_total:
            print(f"\n" + "="*60)
            print(f"[Rank 1] SUCCESS! Data matches perfectly!")
        else:
            print(f"[Rank 1] FAILED - Data mismatch!")
            first_wrong_idx = (result_cpu == 0).nonzero(as_tuple=True)[0]
            if len(first_wrong_idx) > 0:
                idx = first_wrong_idx[0].item()
                print(f"[Rank 1]   First wrong at index {idx}")
                print(f"[Rank 1]   Expected: {idx}")
                print(f"[Rank 1]   Got: {local_buffer[idx].item()}")
            sys.exit(1)
    
    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

