#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import csv
import argparse
from pathlib import Path
import torch
import triton
import triton.language as tl
import iris
from iris._mpi_helpers import mpi_allgather
from examples.common.utils import read_realtime


@triton.jit()
def load_remote(
    data,
    n_elements,
    skip,
    niter,
    curr_rank,
    peer_rank,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    data_mask = offsets < n_elements
    time_stmp_mask = offsets < BLOCK_SIZE

    for i in range(niter + skip):
        if i == skip:
            start = read_realtime()
            tl.store(mm_begin_timestamp_ptr + peer_rank * BLOCK_SIZE + offsets, start, time_stmp_mask)

        # iris.load(data + offsets, curr_rank, peer_rank,heap_bases, data_mask)
        from_base = tl.load(heap_bases + curr_rank)
        to_base = tl.load(heap_bases + peer_rank)
        offset = tl.cast(data + offsets, tl.uint64) - from_base
        translated_ptr = tl.cast(tl.cast(to_base, tl.pointer_type(tl.int8)) + offset, (data + offsets).dtype)
        result = tl.load(translated_ptr, mask=data_mask, cache_modifier=".cv", volatile=True)

    stop = read_realtime()
    tl.store(mm_end_timestamp_ptr + peer_rank * BLOCK_SIZE + offsets, stop, time_stmp_mask)


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    dtype_map = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "int32": torch.int32,
    }
    try:
        return dtype_map[datatype]
    except KeyError:
        raise ValueError(f"Unknown datatype: {datatype}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latency ping-pong benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--datatype",
        type=str,
        default="int32",
        choices=["int8", "fp16", "bf16", "fp32", "int32"],
        help="Datatype for the message payload",
    )
    parser.add_argument(
        "-p",
        "--heap_size",
        type=int,
        default=1 << 32,
        help="Iris heap size",
    )
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=1,
        help="Block size",
    )
    parser.add_argument(
        "-z",
        "--buffer_size",
        type=int,
        default=1,
        help="Length of the source buffer (elements)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "-w",
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup (skip) iterations",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Optional output filename (if omitted, prints results to terminal). Supports .json, .csv",
    )
    return vars(parser.parse_args())


def _pretty_print_matrix(latency_matrix: torch.Tensor) -> None:
    num_ranks = latency_matrix.shape[0]
    col_width = 12
    header = "SRC\\DST".ljust(col_width) + "".join(f"{j:>12}" for j in range(num_ranks))
    print("\nLatency matrix (ns per iter):")
    print(header)
    for i in range(num_ranks):
        row = f"R{i}".ljust(col_width)
        for j in range(num_ranks):
            row += f"{latency_matrix[i, j].item():12.6f}"
        print(row)


def _write_csv(path: Path, latency_matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        num_ranks = latency_matrix.shape[0]
        writer.writerow([""] + [f"R{j}" for j in range(num_ranks)])
        for i in range(num_ranks):
            row = [f"R{i}"] + [f"{latency_matrix[i, j].item():0.6f}" for j in range(num_ranks)]
            writer.writerow(row)


def _write_json(path: Path, latency_matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_ranks = latency_matrix.shape[0]
    rows = []
    for s in range(num_ranks):
        for d in range(num_ranks):
            rows.append(
                {
                    "source_rank": int(s),
                    "destination_rank": int(d),
                    "latency_ns": float(latency_matrix[s, d].item()),
                }
            )
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def save_results(latency_matrix: torch.Tensor, out: str | None) -> None:
    if out is None:
        _pretty_print_matrix(latency_matrix)
        return

    path = Path(out)
    ext = path.suffix.lower()
    if ext == ".json":
        _write_json(path, latency_matrix)
    elif ext == ".csv":
        _write_csv(path, latency_matrix)
    else:
        raise ValueError(f"Unsupported output file extension: {out}")


def print_run_settings(
    args: dict,
    num_ranks: int,
    dtype: torch.dtype,
    BLOCK_SIZE: int,
    BUFFER_LEN: int,
) -> None:
    elem_size = torch.tensor([], dtype=dtype).element_size()
    heap_size = args["heap_size"]
    out = args["output_file"]
    header = "=" * 72
    print(header)
    print("Latency benchmark -- run settings")
    print(header)
    print(f"  num_ranks        : {num_ranks}")
    print(f"  iterations       : {args['iter']}  (timed)")
    print(f"  skip (warmup)    : {args['num_warmup']}")
    print(f"  datatype         : {args['datatype']}  (torch dtype: {dtype})")
    print(f"  element size     : {elem_size} bytes")
    print(f"  heap size        : {heap_size} ({hex(heap_size)})")
    print(f"  block size       : {BLOCK_SIZE}")
    print(f"  buffer len       : {BUFFER_LEN} elements")
    print(f"  output target    : {'<terminal>' if out is None else out}")
    print(header)


if __name__ == "__main__":
    args = parse_args()
    dtype = torch_dtype_from_str(args["datatype"])
    heap_size = args["heap_size"]

    shmem = iris.iris(heap_size)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    BLOCK_SIZE = args["block_size"]
    BUFFER_LEN = args["buffer_size"]

    niter = args["iter"]
    skip = args["num_warmup"]

    if cur_rank == 0:
        print_run_settings(args, num_ranks, dtype, BLOCK_SIZE, BUFFER_LEN)
    shmem.barrier()
    try:
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
    except Exception:
        device_name = "unknown CUDA device"
    print(f"[rank {cur_rank}] ready, device[{device_idx}]: {device_name}")

    mm_begin_timestamp = torch.zeros((num_ranks, BLOCK_SIZE), dtype=torch.int64, device="cuda")
    mm_end_timestamp = torch.zeros((num_ranks, BLOCK_SIZE), dtype=torch.int64, device="cuda")

    local_latency = torch.zeros((num_ranks), dtype=torch.float32, device="cuda")

    source_buffer = shmem.ones(BUFFER_LEN, dtype=dtype)

    grid = lambda meta: (1,)
    for source_rank in range(num_ranks):
        for destination_rank in range(num_ranks):
            if cur_rank == source_rank:
                load_remote[grid](
                    source_buffer,
                    BUFFER_LEN,
                    skip,
                    niter,
                    cur_rank,
                    destination_rank,
                    BLOCK_SIZE,
                    heap_bases,
                    mm_begin_timestamp,
                    mm_end_timestamp,
                )
            shmem.barrier()

    mm_begin_cpu = mm_begin_timestamp.cpu().numpy()
    mm_end_cpu = mm_end_timestamp.cpu().numpy()

    gpu_freq = iris.hip.get_wall_clock_rate(cur_rank)

    for destination_rank in range(num_ranks):
        delta = mm_end_cpu[destination_rank, :] - mm_begin_cpu[destination_rank, :]
        avg_cc = float(delta.sum() / max(1, delta.size) / max(1, niter))
        local_latency[destination_rank] = avg_cc * 1e6 / gpu_freq

    latency_matrix = mpi_allgather(local_latency.cpu())

    if cur_rank == 0:
        save_results(latency_matrix, args["output_file"])
        print("Benchmark complete.")
