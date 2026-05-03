# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for scatter collective communication.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ccl.utils import inline_device_barrier


@triton.jit()
def persistent_scatter(
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
    src: tl.constexpr,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    INLINE_BARRIER: tl.constexpr = False,
):
    """
    Persistent scatter kernel.

    Only the root rank (src) does work. For each destination rank i, it loads
    chunk input[i*M:(i+1)*M, :] and writes to rank i's output tensor.
    Non-root ranks are idle (the kernel is still launched but does no work).

    Args:
        input_ptr: Pointer to input tensor on root (world_size * M, N), unused on non-root
        output_ptr: Pointer to output tensor (M, N)
        M: Number of rows per rank in output
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1)
        iris_rank: Rank in the iris context, used for iris RMA operations
        world_size: Total number of ranks in the group
        rank_start: First iris rank in the group
        rank_stride: Stride between consecutive iris ranks in the group
        src: Source (root) rank within the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    is_src = group_rank == src

    # Only root rank does data movement work; non-root ranks skip to barrier
    if is_src:
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

            # Compute output row and column indices (within M x N output)
            rm_base = pid_m * BLOCK_SIZE_M
            rn_base = pid_n * BLOCK_SIZE_N
            rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
            rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            # Output mask (within M x N)
            output_mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Output offset
            output_base_m = rm[:, None] * stride_out_m
            output_base_n = rn[None, :] * stride_out_n
            output_offset = output_base_m + output_base_n
            output_ptr_target = output_ptr + output_offset
            output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            # Send chunk[i] to each destination rank i
            for i in tl.static_range(world_size):
                target_rank = rank_start + i * rank_stride

                # Input rows for rank i: input[i*M:(i+1)*M, :]
                rm_input = rm + i * M
                input_mask = (rm_input[:, None] < (i + 1) * M) & (rn[None, :] < N)
                combined_mask = output_mask & input_mask

                # Load chunk from input
                input_base_m = rm_input[:, None] * stride_in_m
                input_base_n = rn[None, :] * stride_in_n
                input_offset = input_base_m + input_base_n
                input_ptr_source = input_ptr + input_offset
                input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                data = tl.load(input_ptr_source, mask=combined_mask, other=0.0)

                if i == src:
                    # Local destination (root writes to its own output)
                    tl.store(output_ptr_target, data, mask=combined_mask, cache_modifier=".wt")
                else:
                    # Remote destination: use iris.store
                    iris.store(
                        output_ptr_target,
                        data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        mask=combined_mask,
                        hint=(1, BLOCK_SIZE_N),
                    )

    if INLINE_BARRIER:
        inline_device_barrier(
            pid,
            barrier_flags_ptr,
            wg_done_ptr,
            barrier_sense_ptr,
            heap_bases,
            iris_rank,
            world_size,
            rank_start,
            rank_stride,
            COMM_SMS,
        )


_dummy_barrier_cache: dict = {}


def _get_dummy_barrier(device):
    """Return cached dummy barrier tensors for the no-inline-barrier path."""
    if device not in _dummy_barrier_cache:
        import torch

        _dummy_barrier_cache[device] = tuple(torch.zeros(1, dtype=torch.int32, device=device) for _ in range(3))
    return _dummy_barrier_cache[device]


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src,
    config,
    inline_barrier=False,
    barrier_state=None,
):
    """Launch the Triton scatter kernel."""
    M, N = output_tensor.shape[:2]
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    # All ranks have (world_size * M, N) input for symmetric heap alignment
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    if inline_barrier and barrier_state is not None:
        barrier_flags, wg_done, barrier_sense = barrier_state
    else:
        barrier_flags, wg_done, barrier_sense = _get_dummy_barrier(output_tensor.device)

    iris_launch(
        persistent_scatter,
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
        src,
        barrier_flags,
        wg_done,
        barrier_sense,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        inline_barrier,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="scatter",
        rank=rank_global,
        dtype=output_tensor.dtype,
    )
