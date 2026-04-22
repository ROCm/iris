# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective communication primitive for Iris.
Gathers tensors from all ranks and concatenates them along the last dimension.
"""

import triton
import triton.language as tl
import iris
from iris.tracing.kernel_artifacts import iris_launch
from .config import Config
from .utils import extract_group_info

# Conditional import for Gluon
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from iris.experimental.iris_gluon import IrisDeviceCtx

    GLUON_AVAILABLE = True

    try:
        from triton.experimental.gluon.language.amd.gfx1250 import async_copy as gfx1250_async_copy

        GFX1250_ASYNC_AVAILABLE = True
    except ImportError:
        GFX1250_ASYNC_AVAILABLE = False
except ImportError:
    GLUON_AVAILABLE = False
    GFX1250_ASYNC_AVAILABLE = False


@triton.jit()
def persistent_all_gather(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-gather kernel.

    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (world_size * M, N)
        M: Number of rows per rank (output will be world_size * M rows)
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)
    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(tile_id >= 0)
        tl.assume(stride_in_m >= 0)
        tl.assume(stride_in_n >= 0)
        tl.assume(stride_out_m >= 0)
        tl.assume(stride_out_n >= 0)

        # Compute local row and column indices for input tensor
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm_input = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm_input = tl.max_contiguous(tl.multiple_of(rm_input, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Mask for local input bounds
        input_mask = (rm_input[:, None] < M) & (rn[None, :] < N)

        # Compute input offset and load local shard data once
        input_base_m = rm_input[:, None] * stride_in_m
        input_base_n = rn[None, :] * stride_in_n
        input_offset = input_base_m + input_base_n
        input_ptr_source = input_ptr + input_offset
        input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local input data once for this tile
        data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

        # Send local shard data to all destination ranks
        # Each rank's input goes to output[group_rank * M : (group_rank + 1) * M, :] on all ranks
        for i in tl.static_range(world_size):
            target_rank = rank_start + i * rank_stride

            # Compute global output row indices: offset by group_rank * M
            rm_output = rm_input + group_rank * M

            # Output mask: only write where input was valid
            output_mask = (rm_output[:, None] < (group_rank + 1) * M) & (rn[None, :] < N)

            # Combine masks: must be valid in both input and output
            combined_mask = input_mask & output_mask

            # Compute output offset
            output_base_m = rm_output[:, None] * stride_out_m
            output_base_n = rn[None, :] * stride_out_n
            output_offset = output_base_m + output_base_n
            output_ptr_target = output_ptr + output_offset
            output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            if i == group_rank:
                # Local destination (i == group_rank): use direct store
                tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
            else:
                # Remote destination: use iris.store to send data to remote destination
                # Use iris_rank for iris RMA operations (heap_bases indexing)
                iris.store(
                    output_ptr_target,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=combined_mask,
                    hint=(1, BLOCK_SIZE_N),
                )


@triton.jit()
def persistent_all_gather_partitioned(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-gather kernel with rank-partitioned work distribution.

    Each PID is assigned to work on a specific destination rank, and multiple PIDs
    partition the tiles for that rank. This avoids the inner loop over world_size.

    Work distribution:
    - PIDs are partitioned across destination ranks
    - PIDs_per_rank = COMM_SMS // world_size
    - Each group of PIDs handles all tiles for one destination rank
    - Within each rank group, PIDs partition the tiles

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (world_size * M, N)
        M: Number of rows per rank (output will be world_size * M rows)
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication (must be divisible by world_size)
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    # Partition PIDs across destination ranks
    pids_per_rank = COMM_SMS // world_size
    dest_rank_idx = pid // pids_per_rank  # Which destination rank this PID works on (0 to world_size-1)
    pid_in_rank_group = pid % pids_per_rank  # Which PID within the rank group

    # Compute the actual target rank
    target_rank = rank_start + dest_rank_idx * rank_stride

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    # Iterate over tiles with this PID's offset and stride within the rank group
    for tile_id in range(pid_in_rank_group, total_tiles, pids_per_rank):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(tile_id >= 0)
        tl.assume(stride_in_m >= 0)
        tl.assume(stride_in_n >= 0)
        tl.assume(stride_out_m >= 0)
        tl.assume(stride_out_n >= 0)

        # Compute local row and column indices for input tensor
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm_input = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm_input = tl.max_contiguous(tl.multiple_of(rm_input, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Mask for local input bounds
        input_mask = (rm_input[:, None] < M) & (rn[None, :] < N)

        # Compute input offset and load local shard data once
        input_base_m = rm_input[:, None] * stride_in_m
        input_base_n = rn[None, :] * stride_in_n
        input_offset = input_base_m + input_base_n
        input_ptr_source = input_ptr + input_offset
        input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local input data once for this tile
        data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

        # Compute global output row indices: offset by group_rank * M
        rm_output = rm_input + group_rank * M

        # Output mask: only write where input was valid
        output_mask = (rm_output[:, None] < (group_rank + 1) * M) & (rn[None, :] < N)

        # Combine masks: must be valid in both input and output
        combined_mask = input_mask & output_mask

        # Compute output offset
        output_base_m = rm_output[:, None] * stride_out_m
        output_base_n = rn[None, :] * stride_out_n
        output_offset = output_base_m + output_base_n
        output_ptr_target = output_ptr + output_offset
        output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Send to the assigned destination rank
        if dest_rank_idx == group_rank:
            # Local destination: use direct store
            tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
        else:
            # Remote destination: use iris.store to send data to remote destination
            iris.store(
                output_ptr_target,
                data,
                iris_rank,
                target_rank,
                heap_bases,
                mask=combined_mask,
                hint=(1, BLOCK_SIZE_N),
            )


# Gluon all-gather kernel — linear 1D tiling
#
# Treats input as a flat array of numel contiguous elements. Each CU
# processes BLOCK_SIZE-element chunks with stride COMM_SMS * BLOCK_SIZE.
# No 2D tiling, no swizzle groups, no div/mod.
#
# Key optimizations:
#   - Hoisted pointer translation: local_base loaded once outside chunk loop
#   - Traffic shaping: staggered write order avoids memory controller contention
if GLUON_AVAILABLE:

    @gluon.jit
    def persistent_all_gather_gluon(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        numel,
        output_offset,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE: gl.constexpr,
        COMM_SMS: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
        TRACING: gl.constexpr = False,
    ):
        """
        Persistent all-gather kernel using Gluon with linear 1D tiling.

        Treats the input as a flat contiguous buffer of ``numel`` elements.
        Each CU takes BLOCK_SIZE-element chunks with stride COMM_SMS.

        Args:
            IrisDeviceCtx: Gluon device context class for remote memory operations.
            context_tensor: Opaque tensor holding IrisDeviceCtx state.
            input_ptr: Pointer to local input tensor (contiguous, numel elements).
            output_ptr: Pointer to output tensor (contiguous, world_size * numel elements).
            numel: Total number of elements in input (M * N).
            output_offset: Element offset into output for this rank's data (group_rank * numel).
            group_rank: This rank's index within the ProcessGroup (0..world_size-1).
            iris_rank: This rank's global index in the iris context (for RMA addressing).
            world_size: Total number of ranks in the group.
            rank_start: First iris rank in the group (for RMA target computation).
            rank_stride: Stride between consecutive iris ranks in the group.
            BLOCK_SIZE: Elements per chunk. Must be a multiple of THREADS_PER_WARP * WARPS_PER_CTA.
            COMM_SMS: Number of CUs used for persistent scheduling.
            THREADS_PER_WARP: Threads per warp/wavefront (64 for AMD, 32 for NVIDIA).
            WARPS_PER_CTA: Number of warps per workgroup. Must match num_warps.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)

        pid = gl.program_id(0)

        ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE // (THREADS_PER_WARP * WARPS_PER_CTA)
        layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        local_base = gl.load(ctx.heap_bases + iris_rank)

        for chunk_start in range(pid * BLOCK_SIZE, numel, COMM_SMS * BLOCK_SIZE):
            idx = gl.arange(0, BLOCK_SIZE, layout=layout)
            offs = chunk_start + idx
            mask = offs < numel

            data = gl.load(input_ptr + offs, mask=mask, other=0.0)

            out_offs = output_offset + offs

            # Traffic-shaped stores: stagger write order per rank
            for rank_idx in range(world_size):
                dest_idx = (group_rank + rank_idx) % world_size
                target_iris_rank = rank_start + dest_idx * rank_stride
                output_ptrs = output_ptr + out_offs

                if dest_idx == group_rank:
                    gl.store(output_ptrs, data, mask=mask, cache_modifier=".wt")
                else:
                    target_base = gl.load(ctx.heap_bases + target_iris_rank)
                    ptr_delta = target_base - local_base
                    output_ptrs_int = tl.cast(output_ptrs, gl.uint64)
                    remote_ptrs_int = output_ptrs_int + ptr_delta
                    remote_ptrs = tl.cast(remote_ptrs_int, output_ptrs.dtype)
                    gl.store(remote_ptrs, data, mask=mask)


if GFX1250_ASYNC_AVAILABLE:

    @gluon.jit
    def persistent_all_gather_gluon_gfx1250(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        numel,
        output_offset,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE: gl.constexpr,
        COMM_SMS: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
        TRACING: gl.constexpr = False,
    ):
        """
        Persistent all-gather using GFX1250 async copy through LDS.

        Same linear 1D tiling as persistent_all_gather_gluon, but all data
        movement goes through LDS via async copy — no VGPRs touch the data.

        Data path per chunk:
            HBM → LDS  (gfx1250_async_copy.global_to_shared)
            LDS → HBM  (gfx1250_async_copy.shared_to_global, local write)
            LDS → XGMI (gfx1250_async_copy.shared_to_global, remote writes)

        Hardware requirement: GFX1250 (RDNA4) with async_copy support.
        """
        ctx = IrisDeviceCtx.initialize(context_tensor, tracing=False)

        pid = gl.program_id(0)

        ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE // (THREADS_PER_WARP * WARPS_PER_CTA)
        layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        # LDS buffer for one chunk — data never touches VGPRs
        dtype: gl.constexpr = input_ptr.dtype.element_ty
        ELEM_SIZE_BYTES: gl.constexpr = dtype.primitive_bitwidth // 8
        smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
        smem = gl.allocate_shared_memory(dtype, [BLOCK_SIZE], layout=smem_layout)

        # Hoist local heap base outside the chunk loop
        local_base = gl.load(ctx.heap_bases + iris_rank)

        for chunk_start in range(pid * BLOCK_SIZE, numel, COMM_SMS * BLOCK_SIZE):
            idx = gl.arange(0, BLOCK_SIZE, layout=layout)
            offs = chunk_start + idx
            mask = offs < numel

            # === ASYNC LOAD: HBM → LDS (register-free) ===
            input_ptrs = input_ptr + offs
            gfx1250_async_copy.global_to_shared(smem, input_ptrs, mask=mask, other=0.0)
            gfx1250_async_copy.commit_group()
            gfx1250_async_copy.wait_group(0)

            out_offs = output_offset + offs
            output_base_ptrs = output_ptr + out_offs

            # === ASYNC STORES: LDS → global/XGMI (register-free) ===
            # Traffic-shaped: stagger write order per rank
            for rank_idx in range(world_size):
                dest_idx = (group_rank + rank_idx) % world_size
                target_iris_rank = rank_start + dest_idx * rank_stride

                if dest_idx == group_rank:
                    # Local: LDS → HBM
                    gfx1250_async_copy.shared_to_global(output_base_ptrs, smem, mask=mask)
                else:
                    # Remote: translate pointer via elem_delta, restore
                    # vectorization hints lost by runtime division.
                    target_base = gl.load(ctx.heap_bases + target_iris_rank)
                    ptr_delta = target_base - local_base
                    elem_delta = ptr_delta // ELEM_SIZE_BYTES
                    remote_ptrs = output_ptr + out_offs + elem_delta
                    remote_ptrs = tl.max_contiguous(tl.multiple_of(remote_ptrs, ELEMS_PER_THREAD), ELEMS_PER_THREAD)
                    gfx1250_async_copy.shared_to_global(remote_ptrs, smem, mask=mask)

            # Wait for all stores before reusing LDS on next chunk
            gfx1250_async_copy.commit_group()
            gfx1250_async_copy.wait_group(0)


def all_gather(
    output_tensor,
    input_tensor,
    ctx,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal all-gather collective operation implementation.

    This function is called internally by ctx.ccl.all_gather().
    Users should use the Iris instance method instead:
        >>> ctx.ccl.all_gather(output_tensor, input_tensor)

    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.

    Args:
        output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
        input_tensor: Input tensor of shape (M, N) - local rank's data to send
        ctx: Iris context
        group: ProcessGroup or None. If None, uses all ranks in `iris` context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Set config.all_gather_variant to choose variant: "persistent" or "partitioned"
    """
    # Use provided config or create default one
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    # Extract group information
    # rank_in_group: position within the ProcessGroup (0, 1, 2, ...) - passed as group_rank to kernel
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)

    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape {expected_output_shape}. "
            f"Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # Choose between Triton and Gluon implementation
    if config.use_gluon and GLUON_AVAILABLE:
        # Check if ctx is Iris Gluon (has get_device_context method)
        if not hasattr(ctx, "get_device_context"):
            raise ValueError("use_gluon=True requires Iris Gluon context. Use iris.experimental.iris_gluon.iris()")

        # Gluon only supports the persistent variant
        if config.all_gather_variant != "persistent":
            raise ValueError(
                f"Gluon all_gather only supports all_gather_variant='persistent', got '{config.all_gather_variant}'."
            )

        # Gluon kernel uses linear 1D tiling: BLOCK_SIZE = block_size_m * block_size_n
        block_size = config.block_size_m * config.block_size_n
        threads_per_cta = config.threads_per_warp * config.num_warps
        if block_size < threads_per_cta:
            raise ValueError(
                f"Gluon all-gather requires block_size_m * block_size_n >= "
                f"threads_per_warp * num_warps ({threads_per_cta}), "
                f"got {config.block_size_m} * {config.block_size_n} = {block_size}."
            )
        if block_size % threads_per_cta != 0:
            raise ValueError(
                f"Gluon all-gather requires block_size_m * block_size_n to be a "
                f"multiple of threads_per_warp * num_warps ({threads_per_cta}), "
                f"got {config.block_size_m} * {config.block_size_n} = {block_size}."
            )

        numel = M * N
        output_offset = rank_in_group * numel

        context_tensor = ctx.get_device_context()
        tracing = getattr(ctx, "tracing", None)
        tracing_enabled = bool(tracing and getattr(tracing, "enabled", False))

        # Detect GFX1250 for register-free async copy path
        import torch

        use_gfx1250_async = False
        if GFX1250_ASYNC_AVAILABLE:
            try:
                arch = torch.cuda.get_device_properties(input_tensor.device).gcnArchName
                use_gfx1250_async = "gfx1250" in arch
            except Exception:
                pass

        gluon_kernel = persistent_all_gather_gluon_gfx1250 if use_gfx1250_async else persistent_all_gather_gluon

        iris_launch(
            gluon_kernel,
            (config.comm_sms,),
            IrisDeviceCtx,
            context_tensor,
            input_tensor,
            output_tensor,
            numel,
            output_offset,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            block_size,
            config.comm_sms,
            config.threads_per_warp,
            config.num_warps,
            tracing_enabled,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            algorithm="all_gather",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )
    else:
        if config.use_gluon and not GLUON_AVAILABLE:
            raise ValueError("Gluon is not available. Install Triton with Gluon support or set use_gluon=False")

        # Validate COMM_SMS divisibility for partitioned variant
        if config.all_gather_variant == "partitioned" and config.comm_sms % world_size != 0:
            raise ValueError(
                f"For all_gather_variant='partitioned', COMM_SMS ({config.comm_sms}) must be divisible by world_size ({world_size}). "
                f"Please adjust config.comm_sms to be a multiple of {world_size}."
            )

        heap_bases = ctx.get_heap_bases()

        # Dispatch to the appropriate kernel based on variant
        if config.all_gather_variant == "persistent":
            kernel_fn = persistent_all_gather
        elif config.all_gather_variant == "partitioned":
            kernel_fn = persistent_all_gather_partitioned
        else:
            raise ValueError(f"Unknown all_gather_variant: {config.all_gather_variant}")

        iris_launch(
            kernel_fn,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
            algorithm="all_gather",
            rank=rank_global,
            dtype=input_tensor.dtype,
        )

    if not async_op:
        ctx.barrier()
