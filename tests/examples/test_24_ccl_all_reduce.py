# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Test for examples/24_ccl_all_reduce/example.py."""

import gc
import importlib.util
from pathlib import Path

import pytest
import torch.distributed as dist

import iris


def _load_example(rel_path):
    path = (Path(__file__).parent / rel_path).resolve()
    spec = importlib.util.spec_from_file_location("example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ccl_all_reduce_example():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(heap_size=2**31)
    try:
        _load_example("../../examples/24_ccl_all_reduce/example.py").run(shmem)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
