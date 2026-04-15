#!/usr/bin/env python3
"""
Cross-warp coalescing probe for the gfx1250 all-gather kernel.

Uses the iris tracing infrastructure to instrument the production
persistent_all_gather_gluon_gfx1250 kernel. Sets IRIS_TRACE_ALLGATHER=1
to enable per-thread address dumping and iris trace event recording.
All tracing code is gated by a constexpr and DCE'd when disabled.

Dumps per-thread store addresses so we can verify that adjacent wavefronts
access adjacent 128B cache lines, enabling the hardware to coalesce pairs
into 256B transactions.

Usage (inside FFM container on alola):
    source /ffm/ffmlite_env.sh
    cd /workspace/iris && pip install -e .
    python3 examples/33_coalescing_probe/coalescing_probe.py \
        [--block_size_m 8] [--block_size_n 256] [--num_warps 4] [--dtype fp32]

    # Multi-rank (requires torchrun):
    torchrun --nproc_per_node=4 examples/33_coalescing_probe/coalescing_probe.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# Enable tracing via env variable BEFORE importing iris
os.environ["IRIS_TRACE_ALLGATHER"] = "1"

import triton
from iris.experimental import iris_gluon as iris_gl
from iris.ccl.config import Config


def run_probe(block_size_m, block_size_n, num_warps, dtype_str):
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[dtype_str]
    elem_bytes = torch.tensor([], dtype=dtype).element_size()

    threads_per_warp = triton.runtime.driver.active.get_current_target().warp_size
    total_elems = block_size_m * block_size_n
    elems_per_thread = total_elems // (threads_per_warp * num_warps)

    # Initialize distributed (required by iris).
    # For single-rank FFM testing, set up env vars for single-process group.
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # Single-rank mode: set env vars if not set by torchrun
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
        # FFM does not support NCCL — always use gloo for process group bootstrap.
        # Iris handles GPU-to-GPU data movement via its own symmetric heap / XGMI.
        backend = "gloo"
        dist.init_process_group(backend=backend)

    heap_size = 1 << 28  # 256MB
    shmem = iris_gl.iris(heap_size)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()

    # Allocate tensors: input [M, N], output [world_size * M, N]
    M = block_size_m * 4  # a few tiles worth of rows
    N = block_size_n * 2  # a couple tiles of columns
    input_tensor = shmem.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank))
    output_tensor = shmem.zeros((num_ranks * M, N), dtype=dtype)
    shmem.barrier()

    # Configure kernel
    config = Config(
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        num_warps=num_warps,
        threads_per_warp=threads_per_warp,
        use_gluon=True,
    )

    shmem.barrier()

    # The all_gather function checks IRIS_TRACE_ALLGATHER env var internally
    from iris.ccl.all_gather import all_gather
    all_gather(output_tensor, input_tensor, shmem, config=config)

    torch.cuda.synchronize()

    # Retrieve the addr_dump stashed by all_gather
    addr_dump = getattr(shmem, "_last_addr_dump", None)
    if addr_dump is None:
        print("ERROR: addr_dump not found — tracing may not have been enabled.", file=sys.stderr)
        print("Make sure IRIS_TRACE_ALLGATHER=1 and running on gfx1250.", file=sys.stderr)
        shmem.barrier()
        del shmem
        return None

    # Analyze first tile (tile_id=0)
    tile_addrs = addr_dump[:total_elems].cpu().numpy()
    base_addr = tile_addrs.min()

    # Reshape to [warps, threads, elems_per_thread] following BlockedLayout order
    addrs_shaped = tile_addrs.reshape(num_warps, threads_per_warp, elems_per_thread)

    results = {
        "config": {
            "block_size_m": block_size_m,
            "block_size_n": block_size_n,
            "threads_per_warp": threads_per_warp,
            "warps_per_cta": num_warps,
            "elems_per_thread": elems_per_thread,
            "total_elems": total_elems,
            "dtype": dtype_str,
            "elem_bytes": elem_bytes,
            "M": M,
            "N": N,
            "num_ranks": num_ranks,
            "rank": rank,
        },
        "base_addr": int(base_addr),
        "addrs_flat": tile_addrs.tolist(),
        "per_warp": [],
    }

    if rank == 0:
        print(f"=== Cross-warp coalescing probe (iris tracing) ===")
        print(f"Config: BLOCK_SIZE_M={block_size_m}, BLOCK_SIZE_N={block_size_n}")
        print(f"  THREADS_PER_WARP={threads_per_warp}, WARPS_PER_CTA={num_warps}")
        print(f"  ELEMS_PER_THREAD={elems_per_thread}, TOTAL_ELEMS={total_elems}")
        print(f"  dtype={dtype_str} ({elem_bytes}B/elem), ranks={num_ranks}")
        print(f"  output.data_ptr()=0x{output_tensor.data_ptr():x}")
        print(f"  base_addr=0x{base_addr:x}")
        print()

    for w in range(num_warps):
        warp_addrs = addrs_shaped[w]
        warp_min = int(warp_addrs.min())
        warp_max = int(warp_addrs.max()) + elem_bytes
        byte_offset_min = warp_min - base_addr
        byte_offset_max = warp_max - base_addr
        cl_min = byte_offset_min // 128
        cl_max = (byte_offset_max - 1) // 128

        warp_info = {
            "warp_id": w,
            "byte_range": [int(byte_offset_min), int(byte_offset_max)],
            "cache_lines_128B": [int(cl_min), int(cl_max)],
            "num_cache_lines": int(cl_max - cl_min + 1),
        }
        results["per_warp"].append(warp_info)

        if rank == 0:
            print(f"Warp {w}: bytes [{byte_offset_min}, {byte_offset_max})"
                  f"  cache lines [{cl_min}, {cl_max}]"
                  f"  ({cl_max - cl_min + 1} lines)")

    if rank == 0:
        print()

    # Check cross-warp adjacency
    all_adjacent = True
    for w in range(num_warps - 1):
        this_last = results["per_warp"][w]["cache_lines_128B"][1]
        next_first = results["per_warp"][w + 1]["cache_lines_128B"][0]
        adjacent = (next_first == this_last + 1)
        if rank == 0:
            symbol = "OK" if adjacent else "GAP"
            print(f"Warp {w}->{w+1}: line {this_last} -> {next_first}  [{symbol}]")
        if not adjacent:
            all_adjacent = False

    if rank == 0:
        print()
        if all_adjacent:
            print("RESULT: Adjacent wavefronts access adjacent cache lines")
            print("  Hardware can coalesce pairs of 128B into 256B transactions")
        else:
            print("RESULT: WARNING -- gaps between warp cache line ranges")

    results["cross_warp_adjacent"] = all_adjacent

    # Check monotonically increasing
    offsets = tile_addrs - base_addr
    monotonic = all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))
    results["monotonically_increasing"] = monotonic
    if rank == 0:
        print(f"\nAddress sequence monotonically increasing: {'YES' if monotonic else 'NO'}")

    # Save results
    output_file = f"coalescing_results_rank{rank}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    if rank == 0:
        print(f"\nResults saved to {output_file}")

    shmem.barrier()
    del shmem
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-warp coalescing probe (iris tracing)")
    parser.add_argument("--block_size_m", type=int, default=8)
    parser.add_argument("--block_size_n", type=int, default=256)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()
    run_probe(args.block_size_m, args.block_size_n, args.num_warps, args.dtype)


if __name__ == "__main__":
    main()
