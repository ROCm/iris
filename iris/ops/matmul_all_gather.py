# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using scatter pattern.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
Then scatters C_local tiles to form the full C (M x N) where M = world_size * M_local.

This is useful for tensor-parallel workloads where outputs need to be gathered.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x

from tritonblas.kernels.stages.algorithms.binary import add_vector
from tritonblas.kernels.stages.algorithms.unary import convert_dtype

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def _fused_matmul_all_gather_kernel(
    A,  # (M_local, K) - each rank's local input
    B,  # (K, N) - replicated across ranks
    C_gathered,  # (M, N) - gathered output (M = M_local * world_size)
    bias_ptr,
    M_local: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm_gathered: tl.constexpr,
    stride_cn_gathered: tl.constexpr,
    stride_bias: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused GEMM + all-gather kernel using scatter pattern.

    Computes local GEMM tile and immediately scatters to all ranks.
    No intermediate buffer needed - direct from registers to remote memory.
    """
    pid = tl.program_id(0)

    # Handle multi-XCD devices
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M_local, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm_gathered > 0)
    tl.assume(stride_cn_gathered > 0)

    acc_dtype = tl.int32 if C_gathered.type.element_ty == tl.int8 else tl.float32

    # Persistent loop over local tiles
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute tile coordinates with swizzling
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Compute row and column indices for local tile
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_local
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        # Compute number of K tiles
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        # GEMM loop over K dimension
        for k_tile_idx in range(0, loop_k):
            k_offset = k_tile_idx * BLOCK_SIZE_K
            rk = k_offset + tl.arange(0, BLOCK_SIZE_K)

            # Load A tile
            A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            a = tl.load(tl.multiple_of(A_ptr, (1, 16)))

            # Load B tile
            B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(tl.multiple_of(B_ptr, (16, 1)))

            # Accumulate
            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # Handle remaining K elements if not evenly divisible
        if not EVEN_K:
            k_offset = loop_k * BLOCK_SIZE_K
            rk = k_offset + tl.arange(0, BLOCK_SIZE_K)
            rk_mask = rk < K

            A_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            a = tl.load(tl.multiple_of(A_ptr, (1, 16)), mask=rk_mask[None, :], other=0.0)

            B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(tl.multiple_of(B_ptr, (16, 1)), mask=rk_mask[:, None], other=0.0)

            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # Add bias if provided using tritonBLAS
        if BIAS:
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M_local, other=0.0)
            acc = add_vector(acc, bias_vector, QUANTIZED=False)

        # Convert to output dtype using tritonBLAS
        c = convert_dtype(acc, C_gathered.type.element_ty)

        # Create DeviceContext and destination TensorView for all-gather
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
        dst_view = iris.x.TensorView(C_gathered, M, N, stride_cm_gathered, stride_cn_gathered)
        tile_obj = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

        # Scatter this tile to all ranks using iris.x.all_gather
        # dim=0 means scatter along M dimension (rows)
        iris.x.all_gather(tile_obj, dst_view, dim=0, ctx=ctx)


def matmul_all_gather_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather (none needed for scatter pattern)."""
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    # No workspace needed for scatter pattern
    return FusedWorkspace(
        operation="matmul_all_gather",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )


def matmul_all_gather(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-gather using scatter pattern.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank has A of shape (M_local, K) where M_local = M / world_size.
    The operation computes C_local = A @ B on each rank and immediately
    scatters the tiles to all ranks (all-gather pattern).

    This is a single-kernel implementation - no intermediate buffer needed.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N) where M = M_local * world_size
        A: Input matrix A of shape (M_local, K)
        B: Input matrix B of shape (K, N)
        bias: Optional bias vector (M_local,)
        async_op: If False, performs barrier at end
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace

    Returns:
        FusedWorkspace object
    """
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Validate problem size against block sizes
    assert M_local >= config.block_size_m, (
        f"M_local ({M_local}) must be >= block_size_m ({config.block_size_m}). "
        f"Use smaller block sizes for small problems."
    )
    assert K >= config.block_size_k, (
        f"K ({K}) must be >= block_size_k ({config.block_size_k}). Use smaller block sizes for small problems."
    )
    assert N >= config.block_size_n, (
        f"N ({N}) must be >= block_size_n ({config.block_size_n}). Use smaller block sizes for small problems."
    )

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_preamble(shmem, A, B, config)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm_gathered, stride_cn_gathered = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M_local
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # Launch single fused kernel
    grid = (num_sms,)
    _fused_matmul_all_gather_kernel[grid](
        A,
        B,
        output_tensor,
        bias_ptr,
        M_local,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm_gathered,
        stride_cn_gathered,
        stride_bias,
        shmem.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_sms,
        config.num_xcds,
        use_bias,
        even_k,
        config.allow_tf32,
    )

    if not async_op:
        shmem.barrier()

    return workspace
