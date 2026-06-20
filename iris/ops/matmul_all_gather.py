# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using scatter pattern.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
Then scatters C_local tiles to form the full C (M x N) where M = world_size * M_local.

This is useful for tensor-parallel workloads where outputs need to be gathered.
"""

import logging
from typing import Optional
import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch

from tritonblas.kernels.stages import chiplet_transform_chunked

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def _fused_matmul_all_gather_kernel(
    A,  # (M_local, K) - each rank's local input
    B,  # (K, N) - replicated across ranks
    C_gathered,  # (M, N) - gathered output (M = M_local * world_size)
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
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
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
    pid = chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    # Persistent loop over local tiles using scheduler
    start = pid
    total = NUM_M_TILES * NUM_TILES_N
    stride = NUM_SMS
    for tile_id in range(start, total, stride):
        # Wave-aware tile assignment (similar to hbm_buffer's group-based assignment)
        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_sz)
        pid_n = (tile_id % num_pid_in_group) // group_sz
        pid_m = min(pid_m, NUM_M_TILES - 1)

        # M and N tile indices
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Initialize accumulator for this tile (must be inside the persistent loop!)
        acc_dtype = tl.int32 if C_gathered.type.element_ty == tl.int8 else tl.float32
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k_block_idx in range(NUM_K_BLOCKS):
            # Load A from selected buffer
            rk = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
            a_ptrs = A + rm.to(tl.int64)[:, None] * stride_am + rk[None, :] * stride_ak
            a = tl.load(a_ptrs)

            # Load B at global K position
            B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(B_ptrs)

            # Accumulate
            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # ==================================================================
        # Write output
        # ==================================================================
        if BIAS:
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M_local, other=0.0)
            acc = acc + bias_val[:, None]

        # Convert to output dtype
        c = acc.to(C_gathered.type.element_ty)

        # Create DeviceContext and destination TensorView for all-gather
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
        dst_view = iris.make_tensor_view(C_gathered, M, N, stride_cm_gathered, stride_cn_gathered)
        tile_obj = iris.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

        # Scatter this tile to all ranks using all_gather
        # dim=0 means scatter along M dimension (rows)
        ctx.all_gather(tile_obj, dst_view, dim=0)


def matmul_all_gather_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    selector=None,
    out_dtype: Optional[torch.dtype] = None,
) -> FusedWorkspace:
    """Prepare selector/config launch metadata for matmul_all_gather."""
    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    selector_fallback = False
    if selector is not None:
        launch = _matmul_all_gather_launch_params(M_local, N, selector, A.device)
        variant = "origami"
    elif config is not None:
        launch = _config_launch_params(M_local, N, K, config, A.device)
        variant = "fused_config"
    else:
        c_dtype = A.dtype if out_dtype is None else out_dtype
        selector = _make_origami_selector(M_local, N, K, A, B, c_dtype)
        launch = _matmul_all_gather_launch_params(M_local, N, selector, A.device)
        if _matmul_all_gather_should_use_config_fallback(launch):
            selector = None
            config = FusedConfig()
            launch = _config_launch_params(M_local, N, K, config, A.device)
            variant = "fused_config"
            selector_fallback = True
        else:
            variant = "origami"

    # No workspace needed for scatter pattern
    ws = FusedWorkspace(
        operation="matmul_all_gather",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        variant=variant,
        prepared=True,
    )
    ws.selector = selector
    ws.config = config
    ws.launch_params = launch
    ws.selector_fallback = selector_fallback
    return ws


def _default_chunk_size(total_tiles: int, group_size_m: int, num_xcds: int) -> int:
    chunk_size = group_size_m * group_size_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    return max(1, chunk_size)


def _make_origami_selector(M_local: int, N: int, K: int, A: torch.Tensor, B: torch.Tensor, C):
    from tritonblas.matmul import _make_matmul_selector

    c_dtype = C.dtype if hasattr(C, "dtype") else C
    return _make_matmul_selector(
        M_local,
        N,
        K,
        A.dtype,
        B.dtype,
        c_dtype,
        A.device,
        streamk=False,
    )


def _selector_active_cus(selector, device: torch.device) -> int:
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        props = torch.cuda.get_device_properties(device)
        active_cus = props.multi_processor_count
    return int(active_cus)


def _matmul_all_gather_launch_params(M_local: int, N: int, selector, device: torch.device) -> dict:
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    num_xcds = selector.num_sms
    if num_xcds <= 0:
        num_xcds = 1

    num_tiles_m = (M_local + block_size_m - 1) // block_size_m
    num_tiles_n = (N + block_size_n - 1) // block_size_n
    total_tiles = num_tiles_m * num_tiles_n

    num_sms = min(_selector_active_cus(selector, device), total_tiles)
    chunk_size = _default_chunk_size(num_sms, group_size_m, num_xcds)

    return {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_xcds": num_xcds,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "total_tiles": total_tiles,
        "num_sms": num_sms,
        "chunk_size": chunk_size,
        "num_stages": getattr(selector, "num_stages", 2),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def _matmul_all_gather_should_use_config_fallback(launch: dict) -> bool:
    """Use the old config path for shallow 256x256 selector cases."""
    return (
        launch["block_size_m"] == 256
        and launch["block_size_n"] == 256
        and launch["block_size_k"] == 64
        and launch["total_tiles"] < 2 * launch["num_sms"]
    )


def _config_launch_params(M_local: int, N: int, K: int, config: FusedConfig, device: torch.device) -> dict:
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    num_xcds = config.num_xcds
    if num_xcds <= 0:
        num_xcds = 1
    chunk_size = max(1, config.chunk_size)

    num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n

    return {
        "block_size_m": config.block_size_m,
        "block_size_n": config.block_size_n,
        "block_size_k": config.block_size_k,
        "group_size_m": config.group_size_m,
        "num_xcds": num_xcds,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "total_tiles": num_tiles_m * num_tiles_n,
        "num_sms": num_sms,
        "chunk_size": chunk_size,
        "num_stages": None,
        "allow_tf32": config.allow_tf32,
    }


def matmul_all_gather(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    selector=None,
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
        selector: Optional pre-built tritonBLAS Origami selector

    Returns:
        FusedWorkspace object
    """
    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    from iris.host.logging.logging import _log_rank

    _log_rank(
        logging.DEBUG,
        "matmul_all_gather: shape=(%d,%d,%d) dtype=%s rank=%d/%d",
        M_local * world_size,
        N,
        K,
        A.dtype,
        rank,
        world_size,
        rank=rank,
        num_ranks=world_size,
    )

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Prepare workspace if missing, stale, or explicitly overridden.
    if (
        workspace is None
        or selector is not None
        or config is not None
        or getattr(workspace, "launch_params", None) is None
    ):
        workspace = matmul_all_gather_preamble(
            shmem,
            A,
            B,
            config=config,
            selector=selector,
            out_dtype=output_tensor.dtype,
        )

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

    launch = workspace.launch_params
    config = getattr(workspace, "config", None)
    if getattr(workspace, "selector", None) is None:
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

    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    block_size_k = launch["block_size_k"]
    group_size_m = launch["group_size_m"]
    num_xcds = launch["num_xcds"]
    num_sms = launch["num_sms"]
    chunk_size = launch["chunk_size"]
    num_tiles_m = launch["num_tiles_m"]
    num_tiles_n = launch["num_tiles_n"]
    even_k = K % block_size_k == 0
    num_k_blocks = (K + block_size_k - 1) // block_size_k

    launch_kwargs = {
        "num_warps": 8,
        "matrix_instr_nonkdim": 16,
    }
    if launch["num_stages"] is not None:
        launch_kwargs["num_stages"] = launch["num_stages"]

    # Launch single fused kernel
    grid = (num_sms,)
    iris_launch(
        _fused_matmul_all_gather_kernel,
        grid,
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
        block_size_m,
        block_size_n,
        block_size_k,
        group_size_m,
        num_sms,
        num_xcds,
        chunk_size,
        num_tiles_m,
        num_tiles_n,
        num_k_blocks,
        use_bias,
        even_k,
        launch["allow_tf32"],
        algorithm="matmul_all_gather",
        rank=rank,
        dtype=A.dtype,
        **launch_kwargs,
    )

    if not async_op:
        shmem.barrier()

    return workspace
