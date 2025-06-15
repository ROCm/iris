#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import triton
import triton.language as tl
import random
import numpy as np

import iris


torch.manual_seed(123)
random.seed(123)


@triton.jit
def store_kernel(
    target_buffer,  # tl.tensor: pointer to target data
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

    # Store chunk to target buffer
    iris.store(
        target_buffer + offsets,
        offsets,
        source_rank,
        destination_rank,
        heap_bases_ptr,
        mask=mask,
    )


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
        default="fp32",
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("-z", "--buffer_size", type=int, default=1 << 32, help="Buffer Size")
    parser.add_argument("-b", "--block_size", type=int, default=512, help="Block Size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-d", "--validate", action="store_true", help="Enable validation output")

    parser.add_argument("-p", "--heap_size", type=int, default=1 << 33, help="Iris heap size")

    return vars(parser.parse_args())


def run_experiment(shmem, args, source_rank, destination_rank, buffer):
    dtype = torch_dtype_from_str(args["datatype"])
    cur_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Allocate source and destination buffers on the symmetric heap

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
    n_elements = buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    def run_experiment():
        if cur_rank == source_rank:
            kk = store_kernel[grid](
                buffer,
                n_elements,
                source_rank,
                destination_rank,
                args["block_size"],
                shmem.get_heap_bases(),
            )

    # Warmup
    run_experiment()
    shmem.barrier()

    triton_ms = iris.do_bench(run_experiment, shmem.barrier)

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
        diff_mask = ~torch.isclose(buffer, expected, atol=1)
        breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

        if not torch.allclose(buffer, expected, atol=1):
            max_diff = (buffer - expected).abs().max().item()
            shmem.log(f"Max absolute difference: {max_diff}")
            for idx in breaking_indices:
                idx = tuple(idx.tolist())
                computed_val = buffer[idx]
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


def print_bandwidth_matrix(matrix, label="Unidirectional STORE bandwidth GiB/s [Remote write]"):
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


def main():
    args = parse_args()

    shmem = iris.Iris(args["heap_size"])
    num_ranks = shmem.get_num_ranks()
    bandwidth_matrix = np.zeros((num_ranks, num_ranks), dtype=np.float32)

    dtype = torch_dtype_from_str(args["datatype"])
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    buffer = shmem.zeros(args["buffer_size"] // element_size_bytes, device="cuda", dtype=dtype)

    for source_rank in range(num_ranks):
        for destination_rank in range(num_ranks):
            bandwidth_gbps = run_experiment(shmem, args, source_rank, destination_rank, buffer)
            bandwidth_matrix[source_rank, destination_rank] = bandwidth_gbps
            shmem.barrier()

    if shmem.get_rank() == 0:
        print_bandwidth_matrix(bandwidth_matrix)


if __name__ == "__main__":
    main()
