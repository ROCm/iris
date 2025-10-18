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


def validate_all_scatter(local_tensor, global_tensor, shmem, atol=1):
    """
    Validate all-scatter operation where each rank's local tensor is scattered to all ranks.

    Args:
        local_tensor: The local tensor on this rank before all-scatter
        global_tensor: The global tensor after all-scatter (should contain contributions from all ranks)
        shmem: Iris shmem object
        atol: Absolute tolerance for comparison

    Returns:
        bool: True if validation passes, False otherwise
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Get dimensions
    M, N = local_tensor.shape

    # The global tensor should have dimensions (M, N * world_size)
    # where each rank's N columns are at offset rank * N
    expected_global_shape = (M, N * world_size)

    if global_tensor.shape != expected_global_shape:
        shmem.error(f"Global tensor shape mismatch: expected {expected_global_shape}, got {global_tensor.shape}")
        return False

    # Check that this rank's contribution is in the correct position
    start_col = rank * N
    end_col = (rank + 1) * N
    local_section = global_tensor[:, start_col:end_col]

    diff_mask = ~torch.isclose(local_section, local_tensor, atol=atol)
    breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

    if not torch.allclose(local_section, local_tensor, atol=atol):
        max_diff = (local_section - local_tensor).abs().max().item()
        shmem.info(f"All-scatter validation: Max absolute difference: {max_diff}")
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            computed_val = local_section[idx]
            expected_val = local_tensor[idx]
            shmem.error(
                f"All-scatter mismatch at rank {rank} section, index {idx}: got={computed_val}, expected={expected_val}"
            )
            break
        return False

    return True
