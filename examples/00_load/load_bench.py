#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import triton
import triton.language as tl
import random
import numpy as np
import json
import iris


torch.manual_seed(123)
random.seed(123)


@triton.jit
def load_kernel(
    source_buffer,  # tl.tensor: pointer to source data
    result_buffer,  # tl.tensor: pointer to result data
    buffer_size,  # int32: total number of elements
    source_rank: tl.constexpr,
    destination_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases_ptr: tl.tensor,  # tl.tensor: pointer to heap bases pointers
):
    pid = tl.program_id(0)

    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Get data from target buffer
    result = iris.load(
        source_buffer + offsets,
        source_rank,
        destination_rank,
        heap_bases_ptr,
        mask=mask,
    )

    # Store data to result buffer
    tl.store(result_buffer + offsets, result, mask=mask)


@triton.jit
def store_kernel(
    result_buffer,  # tl.tensor: pointer to result data
    buffer_size,  # int32: total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < buffer_size
    tl.store(result_buffer + offsets, 0, mask=mask)


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "int8": torch.int8,
        "bf16": torch.bfloat16,
    }
    try:
        return dtype_map[datatype]
    except KeyError:
        print(f"Unknown datatype: {datatype}")
        exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse Message Passing configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("-z", "--buffer_size", type=int, default=1 << 32, help="Buffer Size")
    parser.add_argument("-b", "--block_size", type=int, default=512, help="Block Size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-d", "--validate", action="store_true", help="Enable validation output")

    parser.add_argument("-p", "--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument("-o", "--output_file", type=str, default="", help="Output file")
    parser.add_argument("-n", "--num_experiments", type=int, default=10, help="Number of experiments")
    parser.add_argument("-w", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    return vars(parser.parse_args())


def run_experiment(shmem, args, source_rank, destination_rank, source_buffer, result_buffer):
    dtype = torch_dtype_from_str(args["datatype"])
    cur_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if source_rank >= world_size:
        raise ValueError(
            f"Source rank must be less than or equal to the world size. World size is {world_size} and source rank is {source_rank}."
        )
    elif destination_rank >= world_size:
        raise ValueError(
            f"Destination rank must be less than or equal to the world size. World size is {world_size} and destination rank is {destination_rank}."
        )
    if cur_rank == 0:
        if args["verbose"]:
            shmem.log(f"Measuring bandwidth between the ranks {source_rank} and {destination_rank}...")
    n_elements = source_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    def run_store():
        if cur_rank == source_rank:
            store_kernel[grid](result_buffer, n_elements, args["block_size"])

    def run_load():
        if cur_rank == source_rank:
            load_kernel[grid](
                source_buffer,
                result_buffer,
                n_elements,
                source_rank,
                destination_rank,
                args["block_size"],
                shmem.get_heap_bases(),
            )

    # Warmup
    run_store()
    shmem.barrier()
    store_ms = iris.do_bench(run_store, shmem.barrier, n_repeat=args["num_experiments"], n_warmup=args["num_warmup"])

    run_load()
    shmem.barrier()
    get_ms = iris.do_bench(run_load, shmem.barrier, n_repeat=args["num_experiments"], n_warmup=args["num_warmup"])

    # Subtract overhead
    triton_ms = get_ms - store_ms

    bandwidth_gbps = 0
    if cur_rank == source_rank:
        triton_sec = triton_ms * 1e-3
        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        total_bytes = n_elements * element_size_bytes
        bandwidth_gbps = total_bytes / triton_sec / 2**30
        if args["verbose"]:
            shmem.log(f"Copied {total_bytes / 2**30:.2f} GiB in {triton_sec:.4f} seconds")
            shmem.log(f"Bandwidth between {source_rank} and {destination_rank} is {bandwidth_gbps:.4f} GiB/s")
    shmem.barrier()
    bandwidth_gbps = shmem.broadcast(bandwidth_gbps, source_rank)

    success = True
    if args["validate"] and cur_rank == destination_rank:
        if args["verbose"]:
            shmem.log("Validating output...")

        expected = torch.arange(n_elements, dtype=dtype, device="cuda")
        diff_mask = ~torch.isclose(result_buffer, expected, atol=1)
        breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

        if not torch.allclose(result_buffer, expected, atol=1):
            max_diff = (result_buffer - expected).abs().max().item()
            shmem.log(f"Max absolute difference: {max_diff}")
            for idx in breaking_indices:
                idx = tuple(idx.tolist())
                computed_val = result_buffer[idx]
                expected_val = expected[idx]
                shmem.log(f"Mismatch at index {idx}: C={computed_val}, expected={expected_val}")
                success = False
                break

        if success and args["verbose"]:
            shmem.log("Validation successful.")
        if not success and args["verbose"]:
            shmem.log("Validation failed.")

    shmem.barrier()
    return bandwidth_gbps


def print_bandwidth_matrix(matrix, label="Unidirectional LOAD bandwidth GiB/s [Remote read]", output_file=None):
    num_ranks = matrix.shape[0]
    col_width = 10  # Adjust for alignment

    print(f"\n{label}")
    header = " SRC\\DST ".ljust(col_width)
    for dst in range(num_ranks):
        header += f"GPU {dst:02d}".rjust(col_width)
    print(header)

    for src in range(num_ranks):
        row = f"GPU {src:02d}  ->".ljust(col_width)
        for dst in range(num_ranks):
            row += f"{matrix[src, dst]:10.2f}"
        print(row)

    if output_file != "":
        if output_file.endswith(".json"):
            detailed_results = []
            for src in range(num_ranks):
                for dst in range(num_ranks):
                    detailed_results.append(
                        {
                            "source_gpu": f"GPU_{src:02d}",
                            "destination_gpu": f"GPU_{dst:02d}",
                            "source_rank": src,
                            "destination_rank": dst,
                            "bandwidth_gbps": float(matrix[src, dst]),
                        }
                    )
            with open(output_file, "w") as f:
                json.dump(detailed_results, f, indent=2)
        else:
            raise ValueError(f"Unsupported output file extension: {output_file}")


def main():
    args = parse_args()

    shmem = iris.Iris(args["heap_size"])
    num_ranks = shmem.get_num_ranks()
    bandwidth_matrix = np.zeros((num_ranks, num_ranks), dtype=np.float32)

    dtype = torch_dtype_from_str(args["datatype"])
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    source_buffer = shmem.arange(args["buffer_size"] // element_size_bytes, device="cuda", dtype=dtype)
    result_buffer = shmem.zeros_like(source_buffer)

    for source_rank in range(num_ranks):
        for destination_rank in range(num_ranks):
            bandwidth_gbps = run_experiment(shmem, args, source_rank, destination_rank, source_buffer, result_buffer)
            bandwidth_matrix[source_rank, destination_rank] = bandwidth_gbps
            shmem.barrier()

    if shmem.get_rank() == 0:
        print_bandwidth_matrix(bandwidth_matrix, output_file=args["output_file"])


if __name__ == "__main__":
    main()
