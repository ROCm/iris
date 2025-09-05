#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example showing the new PyTorch distributed API for Iris.

This replaces the previous MPI-based examples with PyTorch distributed.
"""

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes


class Iris:
    def __init__(self, heap_size_bytes: int):
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
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank
    )

    iris = Iris(heap_size_bytes)
    print(f"Rank {iris.rank}/{iris.world_size}: Heap base = {hex(iris.heap_base)}")
    print(f"Rank {iris.rank}: Peer heap bases = {[hex(b) for b in iris.peer_heap_bases]}")
    
    dist.barrier()
    dist.destroy_process_group()


def main(nprocs: int = 2, heap_size_bytes: int = 1 << 20):
    init_url = "tcp://127.0.0.1:29500"
    start_processes(
        fn=_worker,
        args=(nprocs, init_url, heap_size_bytes),
        nprocs=nprocs,
        join=True,
    )


if __name__ == "__main__":
    n = torch.cuda.device_count() or 2
    main(nprocs=n, heap_size_bytes=1 << 20)