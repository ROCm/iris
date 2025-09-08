#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import numpy as np
import iris

import importlib.util
from pathlib import Path

current_dir = Path(__file__).parent
file_path = (current_dir / "../../examples/00_load/load_bench.py").resolve()
module_name = "load_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _dist_worker(rank, world_size, fn, *args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12357",
        rank=rank,
        world_size=world_size,
    )
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _run_distributed(fn, num_ranks, *args):
    mp.spawn(_dist_worker, args=(num_ranks, fn, *args), nprocs=num_ranks, join=True)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "buffer_size, heap_size",
    [
        ((1 << 32), (1 << 33)),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_load_bench(request, dtype, buffer_size, heap_size, block_size):
    num_ranks = int(request.config.getoption("--num_ranks"))
    _run_distributed(_test_load_bench_impl, num_ranks, dtype, buffer_size, heap_size, block_size)


def _test_load_bench_impl(rank, world_size, dtype, buffer_size, heap_size, block_size):
    shmem = iris.iris(heap_size)

    bandwidth_matrix = np.zeros((world_size, world_size), dtype=np.float32)
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    source_buffer = shmem.ones(buffer_size // element_size_bytes, dtype=dtype)
    result_buffer = shmem.zeros_like(source_buffer)

    shmem.barrier()

    for source_rank in range(world_size):
        for destination_rank in range(world_size):
            bandwidth_gbps = module.bench_load(
                shmem,
                source_rank,
                destination_rank,
                source_buffer,
                result_buffer,
                block_size,
                dtype,
            )
            bandwidth_matrix[source_rank, destination_rank] = bandwidth_gbps
            shmem.barrier()
