#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: all-to-all with the Gluon Tensor Data Mover (TDM).

Input and output are both (M, N*world_size): input[:, r*N:(r+1)*N] is sent to rank r.

The kernel is inlined here rather than imported from iris.ccl, so this example is
self-contained and does not depend on a TDM path being present in the library.

TDM moves a 2D tile HBM -> LDS and LDS -> HBM/XGMI without the data passing through
registers, so tile size is bounded by LDS rather than by the register file. One
persistent step handles one (destination, tile) pair: load the local slice destined for
that rank, then store it into that rank's output.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse

import torch
import torch.distributed as dist
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

import iris


@gluon.jit
def all_to_all_tdm_kernel(
    input_ptr,
    output_ptr,
    elem_deltas,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """One TDM load and one TDM store per (destination, tile).

    Args:
        input_ptr: Local input, (M, N*world_size).
        output_ptr: Local output, (M, N*world_size).
        elem_deltas: int64[world_size]. Element offset from this rank's symmetric-heap
            base to each peer's, so a peer's output is addressed as
            ``output_ptr + elem_deltas[peer]``.
        M: Rows.
        N: Columns per rank slice.
        stride_in_m, stride_in_n: Input strides.
        stride_out_m, stride_out_n: Output strides.
        group_rank: This rank's index in the group.
        world_size: Ranks in the group.
        BLOCK_M, BLOCK_N: Tile shape. BLOCK_N must divide N -- see the host-side check.
        COMM_SMS: Number of persistent programs.
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    # TDM stores support a single interval padding on the innermost dimension, so with
    # order [1, 0] the padding goes on BLOCK_N.
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], layout=smem_layout)

    n_total = N * world_size
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = N // BLOCK_N
    tiles_per_dest = num_tiles_m * num_tiles_n
    total_steps = world_size * tiles_per_dest

    input_desc = tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, n_total],
        strides=[stride_in_m, stride_in_n],
        block_shape=[BLOCK_M, BLOCK_N],
        layout=smem_layout,
    )

    for step in range(pid, total_steps, COMM_SMS):
        dest = step // tiles_per_dest
        tile = step % tiles_per_dest
        row_off = (tile // num_tiles_n) * BLOCK_M
        tile_n = tile % num_tiles_n

        # Read the slice we owe `dest`; write it into the slice `dest` reserves for us.
        in_col = dest * N + tile_n * BLOCK_N
        out_col = group_rank * N + tile_n * BLOCK_N

        tdm.async_load(input_desc, [row_off, in_col], smem)
        tdm.async_wait(0)

        # Rebuilt per step: the base depends on `dest`, which is a runtime value.
        out_desc = tdm.make_tensor_descriptor(
            base=output_ptr + gl.load(elem_deltas + dest),
            shape=[M, n_total],
            strides=[stride_out_m, stride_out_n],
            block_shape=[BLOCK_M, BLOCK_N],
            layout=smem_layout,
        )
        tdm.async_store(out_desc, [row_off, out_col], smem)
        tdm.async_wait(0)


def _max_lds_bytes():
    """Per-workgroup LDS limit, queried if the driver will say and assumed low if not.

    A wrong-but-conservative ceiling is fine here: the point is that an oversized tile is
    reported against the flag the user typed, not that the bound is exact.
    """
    try:
        import triton.runtime.driver as driver

        limit = int(driver.active.utils.get_device_properties(0)["max_shared_mem"])
        if limit > 0:
            return limit
    except Exception:
        pass
    return 64 * 1024


def all_to_all_tdm(ctx, output_tensor, input_tensor, block_m, block_n, comm_sms, num_warps):
    """Host side: validate the tiling, build the peer offset table, launch.

    Raises:
        ValueError: If the tile shape cannot address the data correctly.
    """
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # Both tile dimensions come straight from the CLI, so check them here: without this
    # a bad value surfaces from inside the layout or the launch, naming neither flag.
    for name, value in (("--block_size_m", block_m), ("--block_size_n", block_n)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a power of 2, got {value}")

    # PaddedSharedLayout pads the inner dimension by 8 elements, so budget for it.
    lds_bytes = block_m * (block_n + 8) * input_tensor.element_size()
    lds_limit = _max_lds_bytes()
    if lds_bytes > lds_limit:
        raise ValueError(
            f"Tile needs {lds_bytes} bytes of LDS, limit is {lds_limit}. "
            f"Reduce --block_size_m ({block_m}) or --block_size_n ({block_n})."
        )

    M, total_n = input_tensor.shape[:2]
    if total_n % world_size:
        raise ValueError(f"Input width {total_n} must be divisible by world_size {world_size}")
    N = total_n // world_size

    # A tile is addressed within one destination's column slice, but the descriptor spans
    # the whole tensor -- so out-of-range columns are clipped at the tensor edge, not at
    # the slice boundary. If BLOCK_N does not divide N the final tile of each slice runs
    # into the neighbouring destination's columns, reading and writing the wrong rank's
    # data with no fault. Refuse instead.
    if N % block_n:
        raise ValueError(
            f"BLOCK_N ({block_n}) must divide the per-rank slice width N ({N}); "
            f"a partial tile would overrun into the next rank's columns."
        )

    # Element offsets to each peer's symmetric heap. Built on the host and copied once:
    # assigning element-wise into a device tensor is world_size separate launches.
    heap_bases = ctx.get_heap_bases()
    elem_size = input_tensor.element_size()
    local_base = int(heap_bases[rank])
    deltas = [(int(heap_bases[r]) - local_base) // elem_size for r in range(world_size)]
    elem_deltas = ctx.zeros(world_size, dtype=torch.int64)
    elem_deltas.copy_(torch.tensor(deltas, dtype=torch.int64))

    all_to_all_tdm_kernel[(comm_sms,)](
        input_tensor,
        output_tensor,
        elem_deltas,
        M,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        rank,
        world_size,
        block_m,
        block_n,
        comm_sms,
        num_warps=num_warps,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="All-to-all using the Gluon Tensor Data Mover",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=512, help="Number of rows")
    parser.add_argument("-n", type=int, default=128, help="Number of columns per rank slice")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--block_size_m", type=int, default=32, help="Tile rows")
    parser.add_argument("--block_size_n", type=int, default=128, help="Tile columns; must divide -n")
    parser.add_argument("--comm_sms", type=int, default=64, help="Number of persistent programs")
    parser.add_argument("--num_warps", type=int, default=4, help="Number of warps")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate against torch.distributed")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args["datatype"]]

    ctx = iris.iris(args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, N = args["m"], args["n"]

    # Distinct per (source, destination) so a slice that arrives from the wrong sender,
    # or carries the wrong destination's payload, is identifiable rather than plausible.
    input_tensor = ctx.zeros((M, N * world_size), dtype=dtype)
    for dest in range(world_size):
        input_tensor[:, dest * N : (dest + 1) * N] = float(rank * 16 + dest)
    output_tensor = ctx.zeros((M, N * world_size), dtype=dtype)

    ctx.barrier()
    all_to_all_tdm(
        ctx,
        output_tensor,
        input_tensor,
        args["block_size_m"],
        args["block_size_n"],
        args["comm_sms"],
        args["num_warps"],
    )
    torch.cuda.synchronize()
    ctx.barrier()

    if args["validate"]:
        expected = torch.empty_like(input_tensor)
        dist.all_to_all_single(expected, input_tensor)
        torch.cuda.synchronize()

        got, want = output_tensor.cpu(), expected.cpu()
        if torch.equal(got, want):
            print(f"[rank {rank}] PASS")
        else:
            print(f"[rank {rank}] FAIL")
            # An all-to-all only moves bytes, so report which slice diverged rather than
            # a norm: the source rank it came from is the diagnostic.
            for src in range(world_size):
                sl = slice(src * N, (src + 1) * N)
                if not torch.equal(got[:, sl], want[:, sl]):
                    print(
                        f"[rank {rank}]   slice from rank {src}: "
                        f"got {got[0, src * N].item()} want {want[0, src * N].item()}"
                    )
            raise SystemExit(1)
    elif rank == 0:
        print(f"all-to-all TDM: {world_size} ranks, {M}x{N} per slice, {args['datatype']}")


if __name__ == "__main__":
    main()
