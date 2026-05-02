# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for point-to-point send/recv communication.

Send kernel: load local data, iris.store to destination buffer, signal flag.
Recv kernel: spin on flag, data already in place from sender's iris.store.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch


@triton.jit()
def send_kernel(
    input_ptr,
    output_ptr,
    flag_ptr,
    done_counter_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    dst_iris_rank: tl.constexpr,
    tag: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """
    Send kernel: write local data to receiver's output buffer, then signal.

    Uses a completion counter so the last PID to finish its tiles
    signals the receiver. This guarantees all remote stores have been
    issued before the flag is set.
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

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        input_ptrs = input_ptr + input_offset
        input_ptrs = tl.multiple_of(input_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        data = tl.load(input_ptrs, mask=mask, other=0.0)

        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        output_ptrs = output_ptr + output_offset
        output_ptrs = tl.multiple_of(output_ptrs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        iris.store(
            output_ptrs,
            data,
            iris_rank,
            dst_iris_rank,
            heap_bases,
            mask=mask,
            hint=(1, BLOCK_SIZE_N),
        )

    # Grid-wide completion: each PID increments counter after finishing all tiles.
    # The last PID (counter reaches COMM_SMS) signals the receiver.
    count = tl.atomic_add(done_counter_ptr, 1, sem="release", scope="sys")
    if count == COMM_SMS - 1:
        iris.atomic_xchg(
            flag_ptr,
            1,
            iris_rank,
            dst_iris_rank,
            heap_bases,
            sem="release",
            scope="sys",
        )


@triton.jit()
def recv_kernel(
    flag_ptr,
    iris_rank: tl.constexpr,
):
    """
    Recv kernel: spin on flag until sender signals completion.

    The receiver's output buffer is already populated by the sender's
    iris.store calls. This kernel only needs to wait for the completion
    flag, then reset it for the next send/recv round.
    """
    while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire", scope="sys") != 1:
        pass

    tl.atomic_xchg(flag_ptr, 0, sem="release", scope="sys")


def launch_send(
    input_tensor,
    output_tensor,
    flag_tensor,
    ctx,
    rank_global,
    dst_iris_rank,
    tag,
    config,
):
    """Launch the Triton send kernel."""
    import torch

    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()

    # Completion counter for grid-wide synchronization (local, not on heap)
    done_counter = torch.zeros(1, dtype=torch.int32, device=input_tensor.device)

    iris_launch(
        send_kernel,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        flag_tensor,
        done_counter,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        heap_bases,
        rank_global,
        dst_iris_rank,
        tag,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="send",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )


def launch_recv(
    flag_tensor,
    rank_global,
    config,
):
    """Launch the Triton recv kernel."""
    iris_launch(
        recv_kernel,
        (1,),
        flag_tensor,
        rank_global,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="recv",
        rank=rank_global,
    )
