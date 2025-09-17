# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch


def validate_gemm(A, B, C, shmem, atol=1):
    expected = A @ B
    diff_mask = ~torch.isclose(C, expected, atol=atol)
    breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

    if not torch.allclose(C, expected, atol=atol):
        max_diff = (C - expected).abs().max().item()
        shmem.info(f"Max absolute difference: {max_diff}")
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            computed_val = C[idx]
            expected_val = expected[idx]
            shmem.error(f"Mismatch at index {idx}: C={computed_val}, expected={expected_val}")
            break
        return False

    return True

def validate_gemm_reduce_scatter(A, B, local_C, rank, world_size, shmem, atol=1):
    full_result = torch.mm(A, B)
    
    rows_per_gpu = A.shape[0] // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + local_C.shape[0]
    
    expected_local = full_result[start_row:end_row, :]
    
    return torch.allclose(local_C, expected_local, atol=atol)
