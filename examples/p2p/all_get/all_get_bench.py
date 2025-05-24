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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Simple data to store (similar to what we accumulate)
    data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    tl.store(target_buffer + offsets, data, mask=mask)


@triton.jit
def all_get_kernel(
    source_buffer,  # tl.tensor: pointer to source data
    target_buffer,  # tl.tensor: pointer to target data
    cur_rank: tl.constexpr,
    buffer_size,  # int32: total number of elements
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases_ptr: tl.tensor,  # tl.tensor: pointer to heap bases pointers
):
    pid = tl.program_id(0)

    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Initialize accumulator
    data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Get chunk from source buffer
    for source_rank in range(world_size):
        if source_rank != cur_rank:  # Skip local HBM access
            remote_data = iris.get(
                source_buffer + offsets,
                cur_rank,
                source_rank,
                heap_bases_ptr,
                mask=mask,
            )
            data += remote_data

    tl.store(target_buffer + offsets, data, mask=mask)


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
    parser.add_argument(
        "-m", "--buffer_size_min", type=int, default=1 << 20, help="Minimum buffer size"
    )
    parser.add_argument(
        "-M", "--buffer_size_max", type=int, default=1 << 32, help="Maximum buffer size"
    )
    parser.add_argument("-b", "--block_size", type=int, default=512, help="Block Size")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--validate", action="store_true", help="Enable validation output"
    )

    parser.add_argument(
        "-p", "--heap_size", type=int, default=1 << 36, help="Iris heap size"
    )

    return vars(parser.parse_args())


def run_experiment(shmem, args, buffer):
    dtype = torch_dtype_from_str(args["datatype"])
    cur_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if args["verbose"]:
        shmem.log(f"Measuring bandwidth for rank {cur_rank} and buffer size {buffer.numel()} elements ({buffer.numel() * torch.tensor([], dtype=dtype).element_size() / 2**30:.2f} GiB)...")
    n_elements = buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Create source and target buffers
    source_buffer = buffer
    target_buffer = shmem.zeros_like(buffer)

    def run_all_get():
        all_get_kernel[grid](
            source_buffer,
            target_buffer,
            cur_rank,
            n_elements,
            world_size,
            args["block_size"],
            shmem.get_heap_bases(),
        )

    def run_store_only():
        store_kernel[grid](
            target_buffer,
            n_elements,
            args["block_size"],
        )

    # Warmup both kernels
    run_all_get()
    run_store_only()
    shmem.barrier()

    # Measure all_get + store
    total_ms = iris.do_bench(run_all_get, shmem.barrier)

    # Measure store overhead
    store_ms = iris.do_bench(run_store_only, shmem.barrier)

    # Net time for just the all_get operations
    net_ms = total_ms - store_ms

    if args["verbose"]:
        shmem.log(f"Total time: {total_ms:.4f} ms, Store overhead: {store_ms:.4f} ms, Net all_get time: {net_ms:.4f} ms")

    triton_sec = net_ms * 1e-3
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    # Each rank gets n_elements from (world_size - 1) other ranks
    total_bytes = n_elements * element_size_bytes * (world_size - 1)
    # Total bandwidth is bytes / time
    bandwidth_gbps = total_bytes / triton_sec / 2**30
    if args["verbose"]:
        shmem.log(f"Got {total_bytes / 2**30:.2f} GiB in {triton_sec:.4f} seconds")
        shmem.log(f"Total bandwidth for rank {cur_rank} is {bandwidth_gbps:.4f} GiB/s")

    success = True
    if args["validate"]:
        if args["verbose"]:
            shmem.log(f"Validating output...")

        expected = torch.arange(n_elements, dtype=dtype, device="cuda")
        diff_mask = ~torch.isclose(target_buffer, expected, atol=1)
        breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

        if not torch.allclose(target_buffer, expected, atol=1):
            max_diff = (target_buffer - expected).abs().max().item()
            shmem.log(f"Max absolute difference: {max_diff}")
            for idx in breaking_indices:
                idx = tuple(idx.tolist())
                computed_val = target_buffer[idx]
                expected_val = expected[idx]
                shmem.log(
                    f"Mismatch at index {idx}: C={computed_val}, expected={expected_val}"
                )
                success = False
                break

        if success and args["verbose"]:
            shmem.log("Validation successful.")
        if not success and args["verbose"]:
            shmem.log("Validation failed.")

    shmem.barrier()
    return bandwidth_gbps


def print_bandwidth_matrix(
    bandwidth_data, buffer_sizes, label="Total Bandwidth (GiB/s) vs Buffer Size"
):
    num_ranks = len(bandwidth_data)
    col_width = 12  # Adjust for alignment

    print(f"\n{label}")
    header = "Buffer Size".ljust(col_width)
    for rank in range(num_ranks):
        header += f"GPU {rank:02d}".rjust(col_width)
    print(header)

    for i, size in enumerate(buffer_sizes):
        row = f"{size/1024/1024:.1f}MB".ljust(col_width)
        for rank in range(num_ranks):
            row += f"{bandwidth_data[rank][i]:12.2f}"
        print(row)


def main():
    args = parse_args()

    heap_size = args["heap_size"]
    shmem = iris.Iris(heap_size)
    num_ranks = shmem.get_num_ranks()

    dtype = torch_dtype_from_str(args["datatype"])
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()

    min_buffer_size = args["buffer_size_min"]
    max_buffer_size = args["buffer_size_max"]
    buffer_size = min_buffer_size
    buffer_sizes = []
    while buffer_size <= max_buffer_size:
        buffer_sizes.append(buffer_size)
        buffer_size *= 2

    # Allocate one large buffer that can fit the maximum size
    max_elements = max_buffer_size // element_size_bytes
    buffer = shmem.zeros(max_elements, device="cuda", dtype=dtype)

    # Initialize bandwidth data structure: [rank][buffer_size_index]
    bandwidth_data = [[0.0 for _ in range(len(buffer_sizes))] for _ in range(num_ranks)]

    for buffer_idx, size in enumerate(buffer_sizes):
        # Use a slice of the large buffer for this experiment
        n_elements = size // element_size_bytes
        sub_buffer = buffer[:n_elements]
        bandwidth_gbps = run_experiment(shmem, args, sub_buffer)
        # Store bandwidth for current rank
        cur_rank = shmem.get_rank()
        bandwidth_data[cur_rank][buffer_idx] = bandwidth_gbps
        shmem.barrier()

    # Gather all bandwidth data to rank 0
    for rank in range(num_ranks):
        for buffer_idx in range(len(buffer_sizes)):
            bandwidth_data[rank][buffer_idx] = shmem.broadcast(bandwidth_data[rank][buffer_idx], rank)
        shmem.barrier()

    if shmem.get_rank() == 0:
        print_bandwidth_matrix(bandwidth_data, buffer_sizes)


if __name__ == "__main__":
    main()
