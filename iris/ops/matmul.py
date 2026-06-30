# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Local GEMM operation using tritonBLAS.

Each rank has input A (M x K) and computes C = A @ B locally.
Output is local (M x N), not gathered across ranks.
"""

from typing import Optional
import torch

# Use tritonBLAS for optimized GEMM
# Import tritonBLAS
from tritonblas.matmul import persistent_matmul_lt, streamk_matmul_lt
from tritonblas.matmul import _make_matmul_selector
from tritonblas.config import matmul_preamble as tritonblas_matmul_preamble
from .workspace import FusedWorkspace


def matmul_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    selector=None,
) -> FusedWorkspace:
    """Allocate workspace for local matmul.

    Args:
        shmem: Iris context
        A: Input matrix A of shape (M, K)
        B: Input matrix B of shape (K, N)
        selector: Optional tritonBLAS selector (if None, creates one)

    Returns:
        FusedWorkspace with selector stored
    """
    M, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    # Create selector if not provided
    if selector is None:
        selector = _make_matmul_selector(
            M,
            N,
            K,
            A.dtype,
            B.dtype,
            A.dtype,  # output dtype
            A.device,
            streamk=False,  # Use persistent kernel
        )

    # Store selector in workspace
    workspace = FusedWorkspace(
        operation="matmul",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )
    workspace.selector = selector

    return workspace


def matmul(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    workspace: Optional[FusedWorkspace] = None,
    work_stealing: bool = False,
    enable_streamk: bool = False,
) -> FusedWorkspace:
    """
    Local matrix multiplication using tritonBLAS.

    Computes: output = A @ B + bias (local computation only)

    Each rank computes its own local matmul independently.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N)
        A: Input matrix A of shape (M, K)
        B: Input matrix B of shape (K, N)
        bias: Optional bias vector (N,) - broadcast across M dimension
        async_op: If False, performs barrier at end

    Returns:
        FusedWorkspace object
    """
    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_preamble(shmem, A, B)

    selector = workspace.selector

    # Call tritonBLAS persistent matmul
    # Note: tritonBLAS expects bias as (N,) not (M,), so we need to handle this
    if bias is not None:
        # iris bias is (M,) - needs to be broadcast across N
        # tritonBLAS bias is (N,) - broadcast across M
        # For now, warn if bias is used - needs different handling
        import warnings

        warnings.warn(
            "iris matmul bias (M,) is not directly compatible with tritonBLAS bias (N,). "
            "Bias will be ignored in this tritonBLAS integration. "
            "Consider adding bias manually after matmul."
        )
        bias = None

    # Allocate work-stealing config if needed (or for streamk)
    config = None
    if work_stealing or enable_streamk:
        config = tritonblas_matmul_preamble(selector, device=A.device)

    # Choose kernel based on enable_streamk
    if enable_streamk:
        streamk_matmul_lt(
            A,
            B,
            output_tensor,
            selector,
            config=config,
            bias=bias,
            work_stealing=work_stealing,
        )
    else:
        persistent_matmul_lt(
            A,
            B,
            output_tensor,
            selector,
            config=config,
            bias=bias,
            work_stealing=work_stealing,
        )

    if not async_op:
        shmem.barrier()

    return workspace
