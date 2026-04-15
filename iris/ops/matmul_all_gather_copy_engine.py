# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using SDMA (copy engine) for scatter.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
Then scatters C_local tiles to form the full C (M x N) where M = world_size * M_local.

This variant uses SDMA hardware for data movement instead of compute shader scatter.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from .config import FusedConfig
from .workspace import FusedWorkspace
from tritonblas.matmul import persistent_matmul_lt, _make_matmul_selector, create_counter_config


@triton.jit()
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit()
def _batch_wait_poster_kernel(
    C_gathered,
    flags,
    flag_iteration,
    M_local,
    N,
    stride_cm,
    stride_cn,
    heap_bases: tl.tensor,
    copy_engine_ctx: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    M_TILES_PER_BATCH: tl.constexpr,
):
    dst_rank = tl.program_id(0)
    if dst_rank >= world_size or dst_rank == cur_rank:
        return

    ptr_dtype = C_gathered.dtype.element_ty
    if ptr_dtype == tl.float16 or ptr_dtype == tl.bfloat16:
        elem_size = 2
    elif ptr_dtype == tl.float32 or ptr_dtype == tl.int32:
        elem_size = 4
    elif ptr_dtype == tl.float64 or ptr_dtype == tl.int64:
        elem_size = 8
    else:
        elem_size = 4

    rows_per_batch = M_TILES_PER_BATCH * BLOCK_SIZE_M
    num_batches = (NUM_M_TILES + M_TILES_PER_BATCH - 1) // M_TILES_PER_BATCH
    rank_row_base = cur_rank * M_local

    for batch_id in range(num_batches):
        tiles_in_batch = tl.minimum(M_TILES_PER_BATCH, NUM_M_TILES - batch_id * M_TILES_PER_BATCH)
        row_offset = batch_id * rows_per_batch
        batch_height = tl.minimum(M_local - row_offset, rows_per_batch)
        wait_value = (flag_iteration + 1) * tiles_in_batch * NUM_TILES_N

        src_ptr = C_gathered + (rank_row_base + row_offset) * stride_cm
        dst_ptr = C_gathered + (rank_row_base + row_offset) * stride_cm

        iris.wait_then_put_rect(
            src_ptr,
            dst_ptr,
            cur_rank,
            dst_rank,
            heap_bases,
            copy_engine_ctx,
            flags + batch_id,
            wait_value,
            N * elem_size,
            batch_height,
            stride_cm * elem_size,
            stride_cm * elem_size,
        )


@triton.jit()
def _device_quiet_kernel(
    copy_engine_ctx: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    dst_rank = tl.program_id(0)
    if dst_rank >= world_size or dst_rank == cur_rank:
        return
    iris.quiet(copy_engine_ctx, dst_rank)


@triton.jit()
def _fused_matmul_all_gather_copy_engine_kernel(
    A,  # (M_local, K) - each rank's local input
    B,  # (K, N) - replicated across ranks
    C_gathered,  # (M, N) - gathered output (M = M_local * world_size)
    bias_ptr,
    flags,
    M_local,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    heap_bases: tl.tensor,
    copy_engine_ctx: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    M_TILES_PER_BATCH: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_COPY_ENGINE: tl.constexpr,
):
    """
    Fused GEMM + all-gather kernel using SDMA (copy engine) for scatter.

    Computes local GEMM tile, stores to memory, then uses SDMA to scatter
    to all ranks. Per-SM flag synchronization ensures completion.
    """
    pid = tl.program_id(0)

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
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % GROUP_SIZE_M)
        pid_n = (tile_id % num_pid_in_group) // GROUP_SIZE_M
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

        c = acc.to(C_gathered.type.element_ty)

        global_offset = (rm + cur_rank * M_local)[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = ((rm + cur_rank * M_local)[:, None] < M) & (rn[None, :] < N)

        # Store locally first; the poster kernel pre-posts POLL+COPY packets
        # that will consume these rows once the batch counter reaches its target.
        tl.store(C_gathered + global_offset, c, mask=mask, cache_modifier=".wt")
        wait_cnt()
        tl.debug_barrier()

        if USE_COPY_ENGINE:
            batch_id = pid_m // M_TILES_PER_BATCH
            tl.atomic_add(flags + batch_id, 1, scope="gpu", sem="release")
        else:
            for remote_rank in range(world_size):
                if remote_rank != cur_rank:
                    iris.put(
                        C_gathered + global_offset,
                        C_gathered + global_offset,
                        cur_rank,
                        remote_rank,
                        heap_bases,
                        copy_engine_ctx,
                        stride_tm=stride_cm,
                        stride_tn=stride_cn,
                        stride_fm=stride_cm,
                        stride_fn=stride_cn,
                        mask=mask,
                        USE_COPY_ENGINE=False,
                        IS_2D_COPY=True,
                        from_base_ptr=C_gathered,
                        to_base_ptr=C_gathered,
                    )


def matmul_all_gather_copy_engine_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    m_tiles_per_batch: int = 1,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_copy_engine including per-batch flags."""
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    # Calculate number of tiles
    # tritonBLAS auto-selects block sizes, get selector to determine tile counts
    num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n

    num_tiles = num_tiles_m * num_tiles_n

    # Calculate number of batches
    num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch

    ws = FusedWorkspace(
        operation="matmul_all_gather_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

    # Allocate one readiness counter per M-batch.
    ws.locks = shmem.zeros((num_batches,), dtype=torch.int32)

    # Store metadata for later use
    # ws.selector = selector
    ws.num_tiles_m = num_tiles_m
    ws.num_tiles_n = num_tiles_n
    ws.num_batches = num_batches

    return ws


def matmul_all_gather_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    use_copy_engine: bool = True,
    flag_iteration: int = 0,
    m_tiles_per_batch: int = 1,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-gather using SDMA (copy engine) for scatter.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank has A of shape (M_local, K) where M_local = M / world_size.
    The operation computes C_local = A @ B on each rank and uses SDMA hardware
    to scatter the tiles to all ranks (all-gather pattern).

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N) where M = M_local * world_size
        A: Input matrix A of shape (M_local, K)
        B: Input matrix B of shape (K, N)
        bias: Optional bias vector (M_local,)
        async_op: If False, performs barrier at end
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace
        use_copy_engine: If True, use SDMA; if False, use compute shader scatter
        flag_iteration: Launch generation for cumulative batch counters.
                        Batch readiness counters are not reset each iteration;
                        the poster waits for the generation-adjusted target.
        m_tiles_per_batch: Number of M tiles grouped behind one readiness flag

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

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_copy_engine_preamble(shmem, A, B, config, m_tiles_per_batch)

    stride_cm, stride_cn = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M_local
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    # Get metadata from workspace
    num_tiles_m = workspace.num_tiles_m
    num_tiles_n = workspace.num_tiles_n
    num_batches = workspace.num_batches

    selector = workspace.selector
    selector_shape = (
        selector.block_m,
        selector.block_n,
        selector.block_k,
        selector.group_m,
    )
    config_shape = (
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
    )
    if selector_shape != config_shape:
        raise ValueError(
            "all_gather_matmul_copy_engine requires selector/config geometry to match: "
            f"selector(M,N,K,G)=({selector.block_m},{selector.block_n},{selector.block_k},{selector.group_m}) "
            f"!= config(M,N,K,G)=({config.block_size_m},{config.block_size_n},{config.block_size_k},{config.group_size_m})"
        )

    if use_copy_engine:
        poster_grid = (world_size,)
        _batch_wait_poster_kernel[poster_grid](
            output_tensor,
            workspace.locks,
            flag_iteration,
            M_local,
            N,
            stride_cm,
            stride_cn,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            rank,
            world_size,
            selector.block_m,
            num_tiles_m,
            num_tiles_n,
            m_tiles_per_batch,
        )

    # Launch GEMM after poster submission so SDMA can wait autonomously.
    if use_copy_engine:
        if bias is not None:
            import warnings

            warnings.warn(
                "Bias is not yet supported in the tritonBLAS SignalView path for "
                "matmul_all_gather_copy_engine. Ignoring bias for this launch."
            )

        counter_config = create_counter_config(
            workspace.locks,
            map_type="block",
            block_group_m=m_tiles_per_batch,
            block_group_n=num_tiles_n,
        )
        c_local_view = output_tensor[rank * M_local : (rank + 1) * M_local, :]
        persistent_matmul_lt(
            A,
            B,
            c_local_view,
            selector,
            bias=None,
            work_stealing=False,
            counter_config=counter_config,
        )
    else:
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        device = A.device
        num_sms = config.num_sms
        if num_sms is None:
            props = torch.cuda.get_device_properties(device)
            num_sms = props.multi_processor_count

        even_k = K % config.block_size_k == 0
        num_k_blocks = (K + config.block_size_k - 1) // config.block_size_k
        num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
        num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
        grid = (num_sms,)
        _fused_matmul_all_gather_copy_engine_kernel[grid](
            A,
            B,
            output_tensor,
            bias_ptr,
            workspace.locks,
            M_local,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            rank,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.block_size_k,
            config.group_size_m,
            num_sms,
            config.num_xcds,
            num_tiles_m,
            num_tiles_n,
            num_k_blocks,
            m_tiles_per_batch,
            use_bias,
            even_k,
            config.allow_tf32,
            use_copy_engine,
        )

    if not async_op:
        if use_copy_engine:
            _device_quiet_kernel[(world_size,)](
                shmem.get_copy_engine_ctx(),
                rank,
                world_size,
            )
        shmem.barrier()

    return workspace
