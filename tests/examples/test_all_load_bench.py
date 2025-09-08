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
file_path = (current_dir / "../../examples/02_all_load/all_load_bench.py").resolve()
module_name = "all_load_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _dist_worker(rank, world_size, fn, *args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12355",
        rank=rank,
        world_size=world_size,
    )
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _run_distributed(fn, num_ranks, *args):
    mp.spawn(_dist_worker, args=(num_ranks, fn, *args), nprocs=num_ranks, join=True)


# ---------------- tests ----------------


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.float16, torch.bfloat16, torch.float32],
)
@pytest.mark.parametrize(
    "buffer_size, heap_size",
    [
        ((1 << 20), (1 << 30)),  # 1 MiB buffer, 1 GiB heap
        ((1 << 22), (1 << 31)),  # 4 MiB buffer, 2 GiB heap
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_all_load_bench(request, dtype, buffer_size, heap_size, block_size):
    num_ranks = int(request.config.getoption("--num_ranks"))
    _run_distributed(_test_all_load_bench_impl, num_ranks, dtype, buffer_size, heap_size, block_size)


def _test_all_load_bench_impl(rank, world_size, dtype, buffer_size, heap_size, block_size):
    shmem = iris.iris(heap_size)

    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = buffer_size // element_size_bytes
    buffer = shmem.zeros(n_elements, dtype=dtype)

    args = {
        "datatype": _torch_dtype_to_str(dtype),
        "block_size": block_size,
        "active_ranks": world_size,
        "num_warmup": 1,
        "num_experiments": 2,
        "verbose": False,
        "validate": False,
    }

    shmem.barrier()
    bandwidth_gbps = module.run_experiment(shmem, args, buffer)
    shmem.barrier()

    assert isinstance(bandwidth_gbps, float)
    assert bandwidth_gbps >= 0.0


@pytest.mark.parametrize("dtype", [torch.float16])
def test_all_load_bench_with_validation(request, dtype):
    num_ranks = int(request.config.getoption("--num_ranks"))
    _run_distributed(_test_all_load_bench_with_validation_impl, num_ranks, dtype)


def _test_all_load_bench_with_validation_impl(rank, world_size, dtype):
    heap_size = 1 << 30
    buffer_size = 1 << 20
    block_size = 512

    shmem = iris.iris(heap_size)

    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = buffer_size // element_size_bytes
    buffer = shmem.zeros(n_elements, dtype=dtype)

    args = {
        "datatype": _torch_dtype_to_str(dtype),
        "block_size": block_size,
        "active_ranks": world_size,
        "num_warmup": 1,
        "num_experiments": 1,
        "verbose": False,
        "validate": True,
    }

    shmem.barrier()
    bandwidth_gbps = module.run_experiment(shmem, args, buffer)
    shmem.barrier()

    assert isinstance(bandwidth_gbps, float)
    assert bandwidth_gbps >= 0.0


def _torch_dtype_to_str(dtype):
    """Convert torch dtype to string format expected by all_load_bench.py"""
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float32:
        return "fp32"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
