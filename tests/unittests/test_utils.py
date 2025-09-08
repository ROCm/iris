# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import socket
import torch.distributed as dist
import torch.multiprocessing as mp

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _dist_worker(rank, world_size, fn, init_method, *args):
    backend = os.environ.get("TORCH_DIST_BACKEND", "nccl")
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()

def dist_spawn(fn, num_ranks, *args):
    """
    Launch `num_ranks` processes via spawn and run `fn(rank, world_size, *args)`.
    Respects MASTER_ADDR/MASTER_PORT if set; otherwise picks a free local port.
    """
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(_find_free_port()))
    init_method = f"tcp://{master_addr}:{master_port}"
    mp.spawn(_dist_worker, args=(num_ranks, fn, init_method, *args),
             nprocs=num_ranks, join=True)
