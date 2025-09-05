# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch distributed launcher for Iris multi-GPU applications.

This module provides utilities for launching Iris applications using PyTorch distributed
instead of MPI, following the pattern described in the issue.
"""

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes


def create_iris_with_distributed_init(heap_size_bytes: int):
    """
    Create an Iris instance after ensuring distributed is initialized.
    
    This function demonstrates the pattern described in the issue for creating
    an Iris context with PyTorch distributed.
    
    Args:
        heap_size_bytes: Size of heap to allocate
        
    Returns:
        Iris instance (but using the launcher pattern from issue)
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed must be initialized before creating Iris instance")
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Device is cuda:<rank>
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Allocate heap and record 64-bit base pointer
    heap = torch.empty(heap_size_bytes, dtype=torch.int8, device=device)
    heap_base = int(heap.data_ptr())

    # All-gather heap bases
    peer_heap_bases = [0 for _ in range(world_size)]
    dist.all_gather_object(peer_heap_bases, heap_base)
    
    # This demonstrates the pattern from the issue but doesn't replace the real Iris class
    class ExampleIrisContext:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.device = device
            self.heap = heap
            self.heap_base = heap_base
            self.peer_heap_bases = peer_heap_bases
    
    return ExampleIrisContext()


def _worker(local_rank: int, world_size: int, init_url: str, heap_size_bytes: int):
    """
    Worker function that runs on each process.
    
    Args:
        local_rank: Local rank of this process
        world_size: Total number of processes
        init_url: URL for process group initialization
        heap_size_bytes: Size of heap to allocate
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank
    )

    # Use the example pattern from the issue
    iris_ctx = create_iris_with_distributed_init(heap_size_bytes)
    
    # For actual use, you would import and use: from iris import iris
    # iris_instance = iris(heap_size_bytes)
    
    dist.barrier()
    dist.destroy_process_group()


def launch_iris(nprocs: int = 2, heap_size_bytes: int = 1 << 20):
    """
    Launch Iris application with multiple processes using PyTorch distributed.
    
    Args:
        nprocs: Number of processes to launch (default: 2)
        heap_size_bytes: Size of heap for each process (default: 1MB)
    """
    init_url = "tcp://127.0.0.1:29500"
    start_processes(
        fn=_worker,
        args=(nprocs, init_url, heap_size_bytes),
        nprocs=nprocs,
        join=True,
    )


def main():
    """
    Main entry point for launching Iris application.
    """
    n = torch.cuda.device_count() or 2
    launch_iris(nprocs=n, heap_size_bytes=1 << 20)


if __name__ == "__main__":
    main()