# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.ops: High-level API for fused GEMM+CCL operations.

This module provides torch-like interfaces for fused matrix multiplication
and collective communication operations. All operations automatically infer
dimensions, strides, and hardware parameters from input tensors.

Usage:
    >>> import iris
    >>> shmem = iris.iris(heap_size)
    >>>
    >>> # Via shmem.ops namespace (recommended)
    >>> A = shmem.randn((M, K), dtype=torch.float16)
    >>> B = shmem.randn((K, N), dtype=torch.float16)
    >>> output = shmem.zeros((M, N), dtype=torch.float16)
    >>> shmem.ops.matmul_all_reduce(output, A, B)
    >>>
    >>> # Or standalone (requires shmem as first parameter)
    >>> import iris.ops as ops
    >>> ops.matmul_all_reduce(shmem, output, A, B)

Available operations:
    - matmul_all_reduce: GEMM + All-Reduce
    - all_gather_matmul: All-Gather + GEMM
    - matmul_all_gather: GEMM + All-Gather
    - matmul_reduce_scatter: GEMM + Reduce-Scatter
"""

from .config import FusedConfig
from .workspace import FusedWorkspace

# Import operations
# from .matmul import matmul  # Simple single-GPU GEMM - TODO: implement
from .matmul_all_reduce import matmul_all_reduce, matmul_all_reduce_preamble
from .all_gather_matmul import all_gather_matmul, all_gather_matmul_preamble
from .matmul_all_gather import matmul_all_gather
from .matmul_reduce_scatter import matmul_reduce_scatter, matmul_reduce_scatter_preamble


class OpsNamespace:
    """
    Namespace for fused GEMM+CCL operations.

    This class provides a convenient namespace for accessing fused operations
    through the shmem.ops property. It holds a reference to the shmem context
    so operations can access rank information and heap bases.

    Example:
        >>> shmem = iris.iris(heap_size)
        >>> A = shmem.randn((M, K), dtype=torch.float16)
        >>> B = shmem.randn((K, N), dtype=torch.float16)
        >>> output = shmem.zeros((M, N), dtype=torch.float16)
        >>> shmem.ops.matmul_all_reduce(output, A, B)
    """

    def __init__(self, shmem):
        """
        Initialize OpsNamespace with shmem context.

        Args:
            shmem: Iris shmem context
        """
        self._shmem = shmem

    def matmul_all_reduce(self, output_tensor, A, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused matrix multiplication and all-reduce.

        Computes: output = all_reduce(A @ B + bias)

        Args:
            output_tensor: Output tensor (M, N)
            A: Input matrix A (M, K)
            B: Input matrix B (K, N)
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> output = shmem.zeros((M, N), dtype=torch.float16)
            >>> shmem.ops.matmul_all_reduce(output, A, B)
        """
        return matmul_all_reduce(self._shmem, output_tensor, A, B, async_op, config, workspace)

    def all_gather_matmul(self, output_tensor, A_sharded, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused all-gather and matrix multiplication.

        Computes: output = all_gather(A_sharded) @ B + bias

        Args:
            output_tensor: Output tensor (M, N)
            A_sharded: Sharded input matrix (M, K_local)
            B: Input matrix B (K, N) where K = K_local * world_size
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> K_local = K // world_size
            >>> A_sharded = shmem.randn((M, K_local), dtype=torch.float16)
            >>> output = shmem.zeros((M, N), dtype=torch.float16)
            >>> shmem.ops.all_gather_matmul(output, A_sharded, B)
        """
        return all_gather_matmul(self._shmem, output_tensor, A_sharded, B, bias, async_op, config, workspace)

    def matmul_all_gather(self, output_tensor, A, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused matrix multiplication and all-gather.

        Computes: output = all_gather(A @ B + bias) along M dimension

        Args:
            output_tensor: Output tensor (M*world_size, N)
            A: Input matrix A (M, K)
            B: Input matrix B (K, N)
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> M_local = M // world_size
            >>> A = shmem.randn((M_local, K), dtype=torch.float16)
            >>> output = shmem.zeros((M, N), dtype=torch.float16)
            >>> shmem.ops.matmul_all_gather(output, A, B)
        """
        return matmul_all_gather(self._shmem, output_tensor, A, B, bias, async_op, config, workspace)

    def matmul_reduce_scatter(self, output_tensor, A, B, bias=None, async_op=False, config=None, workspace=None):
        """
        Fused matrix multiplication and reduce-scatter.

        Computes: output = reduce_scatter(A @ B + bias) along N dimension

        Args:
            output_tensor: Output tensor (M, N_local) where N_local = N / world_size
            A: Input matrix A (M, K)
            B: Input matrix B (K, N)
            bias: Optional bias vector (M,) or (N,)
            async_op: If False, performs barrier at end
            config: Optional FusedConfig for tuning
            workspace: Optional pre-allocated workspace

        Returns:
            workspace: Updated workspace object

        Example:
            >>> N_local = N // world_size
            >>> output = shmem.zeros((M, N_local), dtype=torch.float16)
            >>> shmem.ops.matmul_reduce_scatter(output, A, B)
        """
        return matmul_reduce_scatter(self._shmem, output_tensor, A, B, bias, async_op, config, workspace)


# Export public API
__all__ = [
    # Configuration
    "FusedConfig",
    "FusedWorkspace",
    # Namespace
    "OpsNamespace",
    # Operations
    "matmul",  # Simple single-GPU GEMM
    "matmul_all_reduce",
    "matmul_all_reduce_preamble",
    "all_gather_matmul",
    "all_gather_matmul_preamble",
    "matmul_all_gather",
    "matmul_reduce_scatter",
    "matmul_reduce_scatter_preamble",
]
