# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM operation using pull pattern.

Each rank has a column-sharded input A_sharded (M x K_local).
This operation computes C = all_gather(A_sharded) @ B by pulling
tiles from remote ranks on-demand during GEMM computation.
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
def _fused_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_local: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bias: tl.constexpr,
    heap_bases: tl.tensor,
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
    """Fused all-gather + GEMM kernel using pull pattern."""
    pid = tl.program_id(0)

    # Handle multi-XCD devices
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # Persistent loop over output tiles
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute tile coordinates with swizzling
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Compute row and column indices
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        # Create DeviceContext and TensorView for gather operations
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)
        src_view = iris.x.TensorView(A_sharded, M, K_local, stride_am, stride_ak)

        # Loop over all ranks to pull and accumulate
        for source_rank_id in range(world_size):
            loop_k_local = tl.cdiv(K_local, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k_local -= 1

            # Loop over K dimension for this rank's shard
            for k_block_idx in range(0, loop_k_local):
                k_offset = k_block_idx * BLOCK_SIZE_K

                # Create tile view for this K block
                tile_k = k_offset // BLOCK_SIZE_K
                k_tile = iris.x.TileView(pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                # Pull A tile from source_rank_id using gather primitive
                a = iris.x.gather(k_tile, src_view, source_rank_id, ctx)

                # Load B tile
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk_global = (source_rank_id * K_local) + rk_local
                B_ptr = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)))

                # Accumulate
                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

            # Handle remaining K elements if not evenly divisible
            if not EVEN_K:
                k_offset = loop_k_local * BLOCK_SIZE_K
                tile_k = k_offset // BLOCK_SIZE_K
                k_tile = iris.x.TileView(pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                # Pull A tile from source_rank_id using gather primitive
                a = iris.x.gather(k_tile, src_view, source_rank_id, ctx)

                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk_global = (source_rank_id * K_local) + rk_local
                rk_global_mask = rk_global < K
                B_ptr = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)), mask=rk_global_mask[:, None], other=0.0)

                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

        # Add bias if provided using tritonBLAS
        if BIAS:
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = add_vector(acc, bias_vector, QUANTIZED=False)

        # Convert to output dtype using tritonBLAS
        c = convert_dtype(acc, C.type.element_ty)

        # Store result (manual for now, tritonBLAS store has issues with our indices)
        C_ptr = (
            C
            + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm
            + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
        )
        mask = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < M) & (
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
        )
        tl.store(C_ptr, c, mask=mask)


def all_gather_matmul_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for all_gather_matmul (none needed for pull pattern)."""
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"

    return FusedWorkspace(
        operation="all_gather_matmul",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        prepared=True,
    )


def all_gather_matmul(
    shmem,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """Fused all-gather and matrix multiplication using pull pattern."""
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Validate problem size against block sizes
    assert M >= config.block_size_m, (
        f"M ({M}) must be >= block_size_m ({config.block_size_m}). Use smaller block sizes for small problems."
    )
    assert K_local >= config.block_size_k, (
        f"K_local ({K_local}) must be >= block_size_k ({config.block_size_k}). "
        f"Use smaller block sizes for small problems."
    )
    assert N >= config.block_size_n, (
        f"N ({N}) must be >= block_size_n ({config.block_size_n}). Use smaller block sizes for small problems."
    )

    if workspace is None:
        workspace = all_gather_matmul_preamble(shmem, A_sharded, B, config)

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A_sharded.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K_local % config.block_size_k == 0

    # Launch single fused kernel
    grid = (num_sms,)
    _fused_all_gather_matmul_kernel[grid](
        A_sharded,
        B,
        output_tensor,
        bias_ptr,
        M,
        N,
        K,
        K_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        shmem.heap_bases,
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
