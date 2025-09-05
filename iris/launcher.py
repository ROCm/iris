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


class Iris:
    """
    Iris context manager that initializes distributed communication and GPU heap.
    
    This replaces the MPI-based initialization with PyTorch distributed.
    """
    def __init__(self, heap_size_bytes: int):
        if not dist.is_initialized():
            raise RuntimeError("PyTorch distributed must be initialized before creating Iris instance")
            
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Device is cuda:<rank>
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")

        # Allocate heap and record 64-bit base pointer
        self.heap = torch.empty(heap_size_bytes, dtype=torch.int8, device=self.device)
        self.heap_base = int(self.heap.data_ptr())

        # All-gather heap bases
        self.peer_heap_bases = [0 for _ in range(self.world_size)]
        dist.all_gather_object(self.peer_heap_bases, self.heap_base)


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

    iris = Iris(heap_size_bytes)
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