# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using host-initiated SDMA with POLL packets.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
The host pre-queues SDMA POLL+COPY packets for scatter, then the device kernel just stores
tiles to local HBM and sets flags to trigger the pre-queued transfers.

This is more efficient than device-initiated SDMA because:
- SDMA queue setup happens once on host (not per-tile)
- Device kernel is lightweight (store + set flag)
- SDMA hardware automatically performs scatter when flags are set
"""

from typing import Optional
import torch
import triton
import triton.language as tl

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace

# Import Tile class from anvil module
try:
    import anvil

    Tile = anvil.Tile
except (ImportError, AttributeError):
    Tile = None  # Will raise error later if needed


@triton.jit()
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit()
def _fused_matmul_all_gather_host_copy_engine_kernel(
    A,  # (M_local, K) - each rank's local input
    B,  # (K, N) - replicated across ranks
    C_gathered,  # (M, N) - gathered output (M = M_local * world_size)
    bias_ptr,
    flags,  # Per-tile flags to trigger SDMA
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
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused GEMM + all-gather kernel using host-initiated SDMA with POLL packets.

    Computes local GEMM tile, stores to local HBM, then sets flag to trigger
    pre-queued SDMA transfers.
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

        # Store to local memory (SDMA will read from here when flag is set)
        tl.store(C_gathered + global_offset, c, mask=mask, cache_modifier=".wt")

        # C_ptrs = C_gathered + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        # c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        # tl.store(C_ptrs, c, mask=c_mask, cache_modifier=".wt")

        # TODO which one is better
        # wait_cnt()
        tl.debug_barrier()

        # ═══════════════════════════════════════════════════════════════════
        # Signal Phase: Set flag to trigger pre-queued SDMA transfers
        # ═══════════════════════════════════════════════════════════════════
        # Set flag for this tile (host has pre-queued POLL packets waiting for this)
        # Use tile_id as the flag index
        # tl.store(flags + tile_id, 1, cache_modifier=".wt")
        tl.atomic_add(flags + tile_id, 1, scope="gpu", sem="release")


def matmul_all_gather_host_copy_engine_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_host_copy_engine including per-tile flags."""
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    # Calculate number of tiles
    num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    num_tiles = num_tiles_m * num_tiles_n

    ws = FusedWorkspace(
        operation="matmul_all_gather_host_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

    # Allocate per-tile flags
    ws.locks = shmem.zeros((num_tiles,), dtype=torch.int32)

    return ws


def matmul_all_gather_host_copy_engine(
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
    Fused matrix multiplication and all-gather using host-initiated SDMA with POLL packets.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank has A of shape (M_local, K) where M_local = M / world_size.
    The host pre-queues SDMA POLL+COPY packets for all tiles and ranks.
    The device kernel computes tiles, stores to local HBM, then sets flags.
    SDMA hardware automatically performs scatter when flags are set.

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

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_host_copy_engine_preamble(shmem, A, B, config)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
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

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # Calculate number of tiles
    num_k_blocks = (K + config.block_size_k -1) // config.block_size_k
    num_tiles_m = (M_local + config.block_size_m - 1) // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    num_tiles = num_tiles_m * num_tiles_n

    # Reset flags before kernel launch
    workspace.locks.zero_()
    torch.cuda.synchronize()
    shmem.barrier()

    # ═══════════════════════════════════════════════════════════════════════
    # Device Phase: Launch kernel to compute GEMM + store + set flags
    # ═══════════════════════════════════════════════════════════════════════
    grid = (num_sms,)
    _fused_matmul_all_gather_host_copy_engine_kernel[grid](
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
        use_bias,
        even_k,
        config.allow_tf32,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Host Phase: Enqueue SDMA POLL+COPY packets for all tiles
    # (While kernel is running in parallel on device)
    # ═══════════════════════════════════════════════════════════════════════
    element_size = output_tensor.element_size()
    anvil_lib = shmem.copy_engines

    # Queue POLL+COPY packets for each tile to each remote rank
    # NOTE: Must use same wave-aware assignment as kernel!
    for tile_id in range(num_tiles):
        # Wave-aware tile assignment (must match kernel logic)
        num_pid_in_group = config.group_size_m * num_tiles_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * config.group_size_m
        first_pid_m = min(first_pid_m, num_tiles_m - 1)
        group_sz = min(num_tiles_m - first_pid_m, config.group_size_m)
        tile_m = first_pid_m + ((tile_id % num_pid_in_group) % group_sz)
        tile_n = (tile_id % num_pid_in_group) // group_sz
        tile_m = min(tile_m, num_tiles_m - 1)

        # Calculate tile bounds
        m_start = tile_m * config.block_size_m
        m_end = min(m_start + config.block_size_m, M_local)
        n_start = tile_n * config.block_size_n
        n_end = min(n_start + config.block_size_n, N)

        tile_height = m_end - m_start
        tile_width = n_end - n_start

        # Create Tile object for 2D sub-window copy
        tile_obj = Tile()
        tile_obj.pid_m = 0  # We'll handle offset in data pointer
        tile_obj.pid_n = 0
        tile_obj.block_m = tile_height
        tile_obj.block_n = tile_width
        tile_obj.elem_size = element_size
        tile_obj.src_stride = stride_cm * element_size  # Row stride in bytes

        # Source data pointer (output tensor at this rank's tile location)
        src_offset = (m_start + rank * M_local) * stride_cm + n_start * stride_cn
        tile_obj.data = output_tensor.data_ptr() + src_offset * element_size

        # For each remote rank, queue POLL+COPY
        for remote_rank in range(world_size):
            if remote_rank != rank:
                # Destination is the same logical position on remote rank
                dst_offset = (m_start + rank * M_local) * stride_cm + n_start * stride_cn
                dst_ptr_local = output_tensor.data_ptr() + dst_offset * element_size

                # Translate local pointer to remote rank's address space
                dst_ptr_remote = shmem.translate(dst_ptr_local, rank, remote_rank)
                dst_stride = stride_cm * element_size  # Row stride in bytes

                # Get flag pointer for this tile
                flag_ptr = workspace.locks.data_ptr() + tile_id * workspace.locks.element_size()

                # Use anvil host API to queue POLL+SUB_WINDOW_COPY for 2D tile
                anvil_lib.host_wait_flag_then_put_tile(
                    rank,
                    remote_rank,
                    0,  # channel_idx
                    flag_ptr,
                    1,  # expected_value
                    tile_obj,
                    dst_ptr_remote,
                    dst_stride,
                )

    # Wait for SDMA to complete (all flags have been set, SDMA transfers should finish)
    # Use anvil quiet to wait for SDMA completion
    # TODO part of async_op ?
    for remote_rank in range(world_size):
        if remote_rank != rank:
            anvil_lib.host_quiet(rank, remote_rank, 0)

    if not async_op:
        torch.cuda.synchronize()
        shmem.barrier()

    return workspace
