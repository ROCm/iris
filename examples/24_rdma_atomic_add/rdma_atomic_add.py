#!/usr/bin/env python3
"""
RDMA Atomic Add Example

Demonstrates RDMA atomic fetch-and-add operations between ranks.
Each rank atomically increments a counter on rank 0.
"""

import os
import sys
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from iris.experimental import iris_rdma


@triton.jit
def atomic_add_kernel(
    counter_ptr,
    result_ptr,
    increment,
    dst_rank: tl.constexpr,
    device_ctx,
):
    """
    Each thread atomically adds its increment to the remote counter.
    Returns the old value before increment.
    """
    pid = tl.program_id(0)
    
    # Only first thread does the atomic add
    if pid == 0:
        # Create a mask for single element operation
        mask = tl.full([1], 1, dtype=tl.int1)
        
        # Atomic add: increment counter on dst_rank, get old value
        iris_rdma.atomic_add(
            result_ptr,      # Where to store old value
            counter_ptr,     # Remote counter location (symmetric heap)
            increment,       # Value to add
            dst_rank,        # Which rank has the counter
            device_ctx,
            mask,
        )


def main():
    dtype = torch.int64  # Atomics require int64/uint64
    
    # Initialize distributed
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device_id = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group(
        backend='nccl',
        device_id=device_id
    )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    print(f"[Rank {rank}/{world_size}] Initialized on {device}")
    
    # Initialize RDMA context
    ctx = iris_rdma.IrisRDMA()
    print(f"[Rank {rank}] Iris RDMA initialized")
    print(f"[Rank {rank}]   - Heap base: {ctx.get_heap_base():#x}")
    print(f"[Rank {rank}]   - Queue ptr: {ctx.get_queue_ptr():#x}")
    
    # Get device context for Triton kernels
    device_ctx = ctx.get_device_context()
    
    # Allocate counter and result buffer in symmetric heap
    counter = ctx.zeros(1, dtype=dtype)  # Shared counter
    result = ctx.zeros(1, dtype=dtype)   # Store old value
    
    ctx.barrier()
    
    # ============================================================
    # Rank 0 atomically increments rank 1's counter
    # ============================================================
    print(f"\n[Rank {rank}] === Testing Atomic Add ===")
    
    if rank == 1:
        print(f"[Rank 1] Initial counter value: {counter[0].item()}")
        print(f"[Rank 1] Waiting for rank 0 to increment...")
    
    ctx.barrier()
    
    # Only rank 0 performs the atomic operation (to avoid local atomic on rank 1)
    if rank == 0:
        increment = 42  # Arbitrary test value
        target_rank = 1
        print(f"[Rank 0] Atomically adding {increment} to rank {target_rank}'s counter...")
        
        # Launch atomic add kernel
        grid = (1,)  # Single thread
        atomic_add_kernel[grid](
            counter,         # Counter location (same offset on all ranks)
            result,          # Where to store old value
            increment,       # Value to add
            dst_rank=target_rank,
            device_ctx=device_ctx,
        )
        
        # Synchronize GPU
        torch.cuda.synchronize()
        
        # Read the old value returned by atomic add
        old_value = result.cpu()[0].item()
        print(f"[Rank 0] Atomic add completed. Old value was: {old_value}")
    
    ctx.barrier()
    
    # ============================================================
    # Rank 1: Verify final counter value
    # ============================================================
    if rank == 1:
        print(f"\n[Rank 1] === Verification ===")
        final_value = counter.cpu()[0].item()
        expected = 42  # Only rank 0 added 42
        
        print(f"[Rank 1] Final counter value: {final_value}")
        print(f"[Rank 1] Expected value: {expected}")
        
        if final_value == expected:
            print("\n" + "="*60)
            print("[Rank 1] SUCCESS! RDMA atomic add worked correctly!")
            print("="*60)
        else:
            print(f"[Rank 1] FAILED - Counter value mismatch!")
            print(f"[Rank 1]   Expected: {expected}")
            print(f"[Rank 1]   Got: {final_value}")
            sys.exit(1)
    
    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

