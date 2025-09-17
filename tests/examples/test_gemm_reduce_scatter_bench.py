#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import iris

import importlib.util
from pathlib import Path
from examples.common.utils import (
    Timestamps,
)
current_dir = Path(__file__).parent
import sys
sys.path.append(str(current_dir / "../../examples/13_gemm_reduce_scatter/"))
sys.path.append(str(current_dir / "../../"))
# Import the matmul wrapper
matmul_path = (current_dir / "../../examples/13_gemm_reduce_scatter/matmul_wrapper.py").resolve()
matmul_spec = importlib.util.spec_from_file_location("matmul_wrapper", matmul_path)
matmul_module = importlib.util.module_from_spec(matmul_spec)
matmul_spec.loader.exec_module(matmul_module)

# Import the validation function
validation_path = (current_dir / "../../examples/common/validation.py").resolve()
validation_spec = importlib.util.spec_from_file_location("validation", validation_path)
validation_module = importlib.util.module_from_spec(validation_spec)
validation_spec.loader.exec_module(validation_module)

@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "m, n, k",
    [
        (64, 64, 64),  # Very small for quick testing
        (128, 128, 128),  # Small
        (256, 256, 256),  # Medium
    ],
)
@pytest.mark.parametrize(
    "BLK_M, BLK_N, BLK_K",
    [
        (32, 32, 16),  # Small blocks
        (64, 64, 32),  # Medium blocks
    ],
)
def test_gemm_reduce_scatter(dtype, m, n, k, BLK_M, BLK_N, BLK_K):
    """Worker function for PyTorch distributed execution."""
    heap_size = 1 << 30
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()

    # GEMM
    datatype = dtype

    assert m % world_size == 0, f"M ({m}) must be divisible by world size ({world_size})."
    assert k % world_size == 0, f"K ({k}) must be divisible by world size ({world_size})."

    A = shmem.randn(m, k, device="cuda", dtype=datatype)
    B = shmem.randn(n, k, device="cuda", dtype=datatype).T
    C = shmem.zeros((m, n), device="cuda", dtype=A.dtype)

    M = m
    N = n
    K = k

    # Splitting
    rows_per_gpu = k // world_size
    k = rows_per_gpu
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    local_B = B[start_row:end_row, :]
    local_A = A[:, start_row:end_row]

    compute_buffer = shmem.zeros((m, n), device="cuda", dtype=A.dtype)
    local_output = shmem.zeros((m // world_size, n), device="cuda", dtype=A.dtype)
    
    total_blocks_M = triton.cdiv(m, BLK_M)
    total_blocks_N = triton.cdiv(n, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N

    tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)

    locks = shmem.zeros((288,), device="cuda", dtype=torch.int32)
    P = shmem.zeros(
        (288, BLK_M * BLK_N),
        device="cuda",
        dtype=torch.float32,
    )
    bias = None
    gemm_stream = torch.cuda.Stream()
    timestamps = Timestamps(num_tiles=total_tiles)

    def preamble():
        shmem.barrier()
        tile_completed.zero_()
        shmem.barrier()
        
    def run_experiment():
        nonlocal local_output
        nonlocal compute_buffer

        shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM + Communication")
        with torch.cuda.stream(gemm_stream):
            local_output = matmul_module.matmul_reduce_scatter.apply(
                local_A,
                local_B,
                compute_buffer,
                local_output,
                bias,
                P,
                locks,
                tile_completed,
                rank,
                world_size,
                288,
                BLK_M,
                BLK_N,
                BLK_K,
                6,
                True,
                1,
                8,
                0,
                16,
                2,
                shmem.get_heap_bases(),
                cu_count,
                False,
                timestamps.mm_begin_timestamp,
                timestamps.mm_end_timestamp,
            )
        torch.cuda.nvtx.range_pop()
        shmem.barrier()

    # Synchronize across all GPUs
    shmem.barrier()
    run_experiment()
    shmem.barrier()
    preamble()
    shmem.barrier()
    
    shmem.info("Validating...")

    matmul_module.matmul_reduce_scatter.set_debug(False)
    # Validate global result
    success = validation_module.validate_gemm_reduce_scatter(A, B, local_output, rank, world_size, shmem, atol=2)
    assert success, (
        f"GEMM reduce-scatter validation failed for dtype={dtype}, m={m}, n={n}, k={k}, BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}"
    )
