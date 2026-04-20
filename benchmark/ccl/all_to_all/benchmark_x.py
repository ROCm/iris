#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark: iris.x.all_to_all (Triton) vs iris.x.all_to_all_gluon (Gluon)

Validates correctness and measures bandwidth for the tile-level all-to-all
primitives across many problem sizes.  Supports assembly dumping for
side-by-side comparison.

Run modes
---------
Single size (validate + benchmark):
    python benchmark_x.py -v -b -m 4096 -n 256 -r 8

Sweep across many problem sizes (recommended):
    python benchmark_x.py -v -b --sweep -r 8 --output_file results.json

Dump generated assembly to files:
    python benchmark_x.py --dump_asm -m 1024 -n 128 -r 8

Generate scatter plot from previous sweep results:
    python plot_x_all_to_all.py results.json
"""

import argparse
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris
import iris.x

GLUON_AVAILABLE = False
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    import iris.experimental.iris_gluon as iris_gl

    GLUON_AVAILABLE = hasattr(iris.x, "all_to_all_gluon")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Problem-size sweep grid
# ---------------------------------------------------------------------------

# (M, N_per_rank) pairs covering small / medium / large / various aspect-ratios.
SWEEP_SIZES = [
    # Small
    (128, 64),
    (256, 64),
    (512, 64),
    # Medium
    (1024, 128),
    (2048, 128),
    (1024, 256),
    (2048, 256),
    # Large
    (4096, 128),
    (4096, 256),
    (4096, 512),
    (8192, 256),
    (8192, 512),
    # Extra-large
    (16384, 128),
    (16384, 256),
]


# ---------------------------------------------------------------------------
# Triton kernel wrapper
# ---------------------------------------------------------------------------


@triton.jit
def _triton_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    N_per_rank: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        tile = iris.x.TileView(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = iris.x.make_tensor_view(input_ptr, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.make_tensor_view(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        iris.x.all_to_all(tile, src_view, dst_view, N_per_rank, ctx)


# ---------------------------------------------------------------------------
# Gluon kernel wrapper
# ---------------------------------------------------------------------------

if GLUON_AVAILABLE:

    @gluon.jit
    def _gluon_kernel(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        M,
        N,
        N_per_rank: gl.constexpr,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        num_pid_n,
        cur_rank: gl.constexpr,
        world_size: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
    ):
        pid = gl.program_id(0)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        iris.x.all_to_all_gluon(
            IrisDeviceCtx,
            context_tensor,
            input_ptr,
            output_ptr,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            pid_m,
            pid_n,
            N_per_rank,
            cur_rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark iris.x all_to_all: Triton vs Gluon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=4096, help="Number of rows (ignored when --sweep)")
    parser.add_argument("-n", type=int, default=256, help="Columns per rank (ignored when --sweep)")
    parser.add_argument("--block_size_m", type=int, default=64, help="BLOCK_SIZE_M")
    parser.add_argument("--block_size_n", type=int, default=256, help="BLOCK_SIZE_N")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run timing loop")
    parser.add_argument("--sweep", action="store_true", help="Sweep across many (M, N) problem sizes")
    parser.add_argument(
        "--dump_asm",
        action="store_true",
        help="Dump generated AMDGCN assembly for Triton and Gluon kernels to .asm files",
    )
    parser.add_argument("--output_file", type=str, default="log_x_all_to_all.json", help="JSON output path")
    parser.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes")
    return vars(parser.parse_args())


def _run_one_size(
    M,
    N,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    dtype,
    shmem,
    rank,
    ws,
    context_tensor,
    args,
):
    """Run validation and/or benchmark for a single (M, N) problem size.

    Returns a dict with timing / bandwidth / validation results, or None on
    the non-zero ranks (data is only collected on rank 0).
    """
    total_N = N * ws
    element_size = torch.tensor([], dtype=dtype).element_size()

    iris_input = shmem.zeros((M, total_N), dtype=dtype)
    iris_output_triton = shmem.zeros((M, total_N), dtype=dtype)
    iris_output_gluon = shmem.zeros((M, total_N), dtype=dtype) if GLUON_AVAILABLE else None

    # Fill input: chunk i is filled with value (rank * 10 + i + 1).
    for target_rank in range(ws):
        iris_input[:, target_rank * N : (target_rank + 1) * N] = float(rank * 10 + target_rank + 1)

    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (total_N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    grid_triton = (total_tiles,)
    grid_gluon = (total_tiles,)

    def run_triton():
        _triton_kernel[grid_triton](
            iris_input,
            iris_output_triton,
            M,
            total_N,
            N,
            iris_input.stride(0),
            iris_input.stride(1),
            iris_output_triton.stride(0),
            iris_output_triton.stride(1),
            context_tensor,
            rank,
            ws,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )

    def run_gluon():
        if not GLUON_AVAILABLE:
            return
        _gluon_kernel[grid_gluon](
            iris_gl.IrisDeviceCtx,
            context_tensor,
            iris_input,
            iris_output_gluon,
            M,
            total_N,
            N,
            iris_input.stride(0),
            iris_input.stride(1),
            iris_output_gluon.stride(0),
            iris_output_gluon.stride(1),
            num_pid_n,
            rank,
            ws,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            num_warps=4,
        )

    result = {
        "M": M,
        "N": N,
        "world_size": ws,
        "dtype": str(dtype).replace("torch.", ""),
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": BLOCK_SIZE_N,
        "total_bytes": (ws - 1) * M * N * element_size,
    }

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------
    if args["validate"]:
        # Build expected output: output[:, src*N:(src+1)*N] = src_rank * 10 + rank + 1
        expected = shmem.zeros((M, total_N), dtype=dtype)
        for src_rank in range(ws):
            expected[:, src_rank * N : (src_rank + 1) * N] = float(src_rank * 10 + rank + 1)

        atol = 0.5

        # Triton
        iris_output_triton.zero_()
        shmem.barrier()
        run_triton()
        torch.cuda.synchronize()
        shmem.barrier()
        ok_triton = torch.allclose(iris_output_triton, expected, atol=atol)
        result["triton_valid"] = bool(ok_triton)

        if rank == 0:
            status = "PASS" if ok_triton else "FAIL"
            print(f"  [Triton]  M={M:6d} N_per_rank={N:5d}: validation {status}")

        # Gluon
        if GLUON_AVAILABLE:
            iris_output_gluon.zero_()
            shmem.barrier()
            run_gluon()
            torch.cuda.synchronize()
            shmem.barrier()
            ok_gluon = torch.allclose(iris_output_gluon, expected, atol=atol)
            result["gluon_valid"] = bool(ok_gluon)

            if rank == 0:
                status = "PASS" if ok_gluon else "FAIL"
                print(f"  [Gluon]   M={M:6d} N_per_rank={N:5d}: validation {status}")

    # -------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------
    if args["benchmark"]:
        total_bytes = (ws - 1) * M * N * element_size
        total_bytes_gb = total_bytes / (1024**3)

        shmem.barrier()
        triton_ms = iris.do_bench(run_triton, shmem.barrier)
        bw_triton = total_bytes_gb / (triton_ms * 1e-3) if triton_ms > 0 else 0.0
        result["triton_ms"] = triton_ms
        result["triton_bandwidth_gbps"] = bw_triton

        if rank == 0:
            print(f"  [Triton]  M={M:6d} N_per_rank={N:5d}: {triton_ms:8.3f} ms  {bw_triton:7.3f} GB/s")

        if GLUON_AVAILABLE:
            shmem.barrier()
            gluon_ms = iris.do_bench(run_gluon, shmem.barrier)
            bw_gluon = total_bytes_gb / (gluon_ms * 1e-3) if gluon_ms > 0 else 0.0
            ratio = (bw_gluon / bw_triton * 100) if bw_triton > 0 else 0.0
            result["gluon_ms"] = gluon_ms
            result["gluon_bandwidth_gbps"] = bw_gluon
            result["gluon_vs_triton_percent"] = ratio

            if rank == 0:
                print(
                    f"  [Gluon]   M={M:6d} N_per_rank={N:5d}: {gluon_ms:8.3f} ms  {bw_gluon:7.3f} GB/s"
                    f"  ({ratio:5.1f}% of Triton)"
                )

    return result


def _dump_assembly(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, dtype, shmem, rank, ws, context_tensor):
    """Compile kernels and dump AMDGCN assembly to text files.

    Files are written only on rank 0.  Both backends are compiled with the
    same problem configuration so the resulting assembly is directly comparable.
    """
    total_N = N * ws
    dummy = shmem.zeros((M, total_N), dtype=dtype)

    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (total_N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n

    # Trigger compilation (warmup run).
    kk_triton = _triton_kernel[(total_tiles,)](
        dummy,
        dummy,
        M,
        total_N,
        N,
        dummy.stride(0),
        dummy.stride(1),
        dummy.stride(0),
        dummy.stride(1),
        context_tensor,
        rank,
        ws,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    if rank == 0 and hasattr(kk_triton, "asm") and "amdgcn" in kk_triton.asm:
        asm = kk_triton.asm["amdgcn"]
        fname = f"triton_all_to_all_M{M}_N{N}_bm{BLOCK_SIZE_M}_bn{BLOCK_SIZE_N}.asm"
        with open(fname, "w") as f:
            f.write(asm)
        n_regs = getattr(kk_triton, "n_regs", "?")
        n_spills = getattr(kk_triton, "n_spills", "?")
        print(f"  [Triton]  {fname}  ({len(asm):,} chars, {n_regs} VGPRs, {n_spills} spills)")

    if GLUON_AVAILABLE:
        kk_gluon = _gluon_kernel[(total_tiles,)](
            iris_gl.IrisDeviceCtx,
            context_tensor,
            dummy,
            dummy,
            M,
            total_N,
            N,
            dummy.stride(0),
            dummy.stride(1),
            dummy.stride(0),
            dummy.stride(1),
            num_pid_n,
            rank,
            ws,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            num_warps=4,
        )
        if rank == 0 and hasattr(kk_gluon, "asm") and "amdgcn" in kk_gluon.asm:
            asm = kk_gluon.asm["amdgcn"]
            fname = f"gluon_all_to_all_M{M}_N{N}_bm{BLOCK_SIZE_M}_bn{BLOCK_SIZE_N}.asm"
            with open(fname, "w") as f:
                f.write(asm)
            n_regs = getattr(kk_gluon, "n_regs", "?")
            n_spills = getattr(kk_gluon, "n_spills", "?")
            print(f"  [Gluon]   {fname}  ({len(asm):,} chars, {n_regs} VGPRs, {n_spills} spills)")

    shmem.barrier()


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=init_url, world_size=world_size, rank=local_rank)

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]

    BLOCK_SIZE_M = args["block_size_m"]
    BLOCK_SIZE_N = args["block_size_n"]

    # Use Gluon-based iris if available (required for Gluon kernel).
    if GLUON_AVAILABLE:
        shmem = iris_gl.iris(args["heap_size"])
    else:
        shmem = iris.iris(args["heap_size"])

    rank = shmem.get_rank()
    ws = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()

    # Determine problem sizes to run.
    sizes = SWEEP_SIZES if args["sweep"] else [(args["m"], args["n"])]

    if rank == 0 and len(sizes) > 1:
        print(f"\n=== iris.x all_to_all sweep  world_size={ws}  dtype={args['datatype']} ===\n")

    all_results = []

    for M, N in sizes:
        if rank == 0 and len(sizes) > 1:
            print(f"--- M={M}, N_per_rank={N} ---")

        if args["dump_asm"]:
            _dump_assembly(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, dtype, shmem, rank, ws, context_tensor)
            continue

        result = _run_one_size(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, dtype, shmem, rank, ws, context_tensor, args)
        all_results.append(result)

    # Save results to JSON (rank 0 only).
    if rank == 0 and all_results and args["output_file"]:
        with open(args["output_file"], "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args['output_file']}")

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]
    init_url = "tcp://127.0.0.1:29572"
    mp.spawn(fn=_worker, args=(num_ranks, init_url, args), nprocs=num_ranks, join=True)


if __name__ == "__main__":
    main()
