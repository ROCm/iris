# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-to-all and all-to-all-v collective operations.
"""

import gc

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size
        (256, 128, 32, 16),  # Minimum BLOCK_N=16
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_to_all(dtype, M, N, block_size_m, block_size_n):
    """Test uniform all-to-all by comparing against torch.distributed."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # Each rank creates distinct data per destination
    # input[:, i*N:(i+1)*N] = data to send to rank i
    pytorch_input = torch.randn(M, N * world_size, dtype=dtype, device=f"cuda:{rank}")
    # Make deterministic: rank r sends (r * world_size + dest) to each dest
    for dest in range(world_size):
        pytorch_input[:, dest * N : (dest + 1) * N] = rank * world_size + dest

    # PyTorch reference: list-based all_to_all
    pytorch_input_list = [pytorch_input[:, i * N : (i + 1) * N].contiguous() for i in range(world_size)]
    pytorch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    ctx.barrier()
    dist.all_to_all(pytorch_output_list, pytorch_input_list)
    torch.cuda.synchronize()

    pytorch_output_concat = torch.cat(pytorch_output_list, dim=1)

    # Iris all-to-all: concatenated format
    iris_input = ctx.zeros((M, N * world_size), dtype=dtype)
    iris_input.copy_(pytorch_input)
    iris_output = ctx.zeros((M, N * world_size), dtype=dtype)

    ctx.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    ctx.ccl.all_to_all(iris_output, iris_input, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    try:
        assert torch.allclose(iris_output, pytorch_output_concat, atol=atol), (
            f"Rank {rank}: max diff = {torch.abs(iris_output - pytorch_output_concat).max().item()}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_to_all_self_copy(dtype):
    """Test all-to-all with world_size=1 equivalent (only self-copy path)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, N = 64, 32
    iris_input = ctx.zeros((M, N * world_size), dtype=dtype)
    # Fill each chunk with the destination rank index
    for i in range(world_size):
        iris_input[:, i * N : (i + 1) * N] = float(i + rank * 100)

    iris_output = ctx.zeros((M, N * world_size), dtype=dtype)

    ctx.barrier()
    ctx.ccl.all_to_all(iris_output, iris_input)
    torch.cuda.synchronize()

    # Verify using torch.distributed reference
    input_list = [iris_input[:, i * N : (i + 1) * N].clone().contiguous() for i in range(world_size)]
    output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_to_all(output_list, input_list)
    torch.cuda.synchronize()

    ref = torch.cat(output_list, dim=1)
    atol = 1e-3 if dtype == torch.float16 else 1e-5

    try:
        assert torch.allclose(iris_output, ref, atol=atol), (
            f"Rank {rank}: max diff = {torch.abs(iris_output - ref).max().item()}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_all_to_all_v_uniform(dtype):
    """Test AllToAllv with uniform counts (should produce same result as uniform all-to-all)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    count_per_rank = 1024
    total_elems = count_per_rank * world_size

    # Uniform: send count_per_rank elements to each rank
    send_counts = [count_per_rank] * world_size
    send_displs = [i * count_per_rank for i in range(world_size)]
    recv_counts = [count_per_rank] * world_size
    recv_displs = [i * count_per_rank for i in range(world_size)]

    # Input: each chunk has deterministic values
    iris_input = ctx.zeros(total_elems, dtype=dtype)
    for i in range(world_size):
        iris_input[i * count_per_rank : (i + 1) * count_per_rank] = rank * world_size + i

    iris_output = ctx.zeros(total_elems, dtype=dtype)

    # Reference via torch.distributed
    ref_input = iris_input.clone()
    ref_input_list = [ref_input[i * count_per_rank : (i + 1) * count_per_rank].contiguous() for i in range(world_size)]
    ref_output_list = [torch.zeros(count_per_rank, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    ctx.barrier()
    dist.all_to_all(ref_output_list, ref_input_list)
    torch.cuda.synchronize()
    ref_output = torch.cat(ref_output_list)

    ctx.barrier()
    ctx.ccl.all_to_all_v(
        iris_output,
        iris_input,
        send_counts,
        send_displs,
        recv_counts,
        recv_displs,
    )
    torch.cuda.synchronize()

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    try:
        assert torch.allclose(iris_output, ref_output, atol=atol), (
            f"Rank {rank}: max diff = {torch.abs(iris_output - ref_output).max().item()}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_all_to_all_v_variable(dtype):
    """Test AllToAllv with variable counts simulating MoE routing imbalance."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # Variable counts: rank i sends (i + j + 1) * 128 elements to rank j
    # This simulates MoE-style routing imbalance
    send_counts = [(rank + j + 1) * 128 for j in range(world_size)]
    recv_counts = [(i + rank + 1) * 128 for i in range(world_size)]

    send_displs = []
    offset = 0
    for c in send_counts:
        send_displs.append(offset)
        offset += c

    recv_displs = []
    offset = 0
    for c in recv_counts:
        recv_displs.append(offset)
        offset += c

    total_send = sum(send_counts)
    total_recv = sum(recv_counts)

    # Symmetric heap requires all ranks to allocate the same size.
    # Use all_reduce to find the max across ranks.
    max_send = torch.tensor([total_send], device=f"cuda:{rank}")
    max_recv = torch.tensor([total_recv], device=f"cuda:{rank}")
    dist.all_reduce(max_send, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_recv, op=dist.ReduceOp.MAX)
    alloc_send = int(max_send.item())
    alloc_recv = int(max_recv.item())

    # Input: fill each chunk with (rank * 1000 + dest)
    iris_input = ctx.zeros(alloc_send, dtype=dtype)
    for j in range(world_size):
        iris_input[send_displs[j] : send_displs[j] + send_counts[j]] = rank * 1000 + j

    iris_output = ctx.zeros(alloc_recv, dtype=dtype)

    # Reference via torch.distributed.all_to_all_single
    ref_input = iris_input[:total_send].clone()
    ref_output = torch.zeros(total_recv, dtype=dtype, device=f"cuda:{rank}")

    ctx.barrier()
    dist.all_to_all_single(
        ref_output,
        ref_input,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    torch.cuda.synchronize()

    ctx.barrier()
    ctx.ccl.all_to_all_v(
        iris_output,
        iris_input,
        send_counts,
        send_displs,
        recv_counts,
        recv_displs,
    )
    torch.cuda.synchronize()

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    try:
        assert torch.allclose(iris_output, ref_output, atol=atol), (
            f"Rank {rank}: max diff = {torch.abs(iris_output - ref_output).max().item()}\n"
            f"send_counts={send_counts}, recv_counts={recv_counts}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_all_to_all_v_empty_chunks(dtype):
    """Test AllToAllv where some ranks send 0 elements to some peers."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    if world_size < 2:
        pytest.skip("Need at least 2 ranks")

    # Rank i only sends to ranks with index > i (upper triangle)
    send_counts = []
    for j in range(world_size):
        if j > rank:
            send_counts.append(256)
        elif j == rank:
            send_counts.append(256)  # self-copy
        else:
            send_counts.append(0)

    recv_counts = []
    for i in range(world_size):
        if rank > i:
            recv_counts.append(256)
        elif rank == i:
            recv_counts.append(256)  # self-copy
        else:
            recv_counts.append(0)

    send_displs = []
    offset = 0
    for c in send_counts:
        send_displs.append(offset)
        offset += c

    recv_displs = []
    offset = 0
    for c in recv_counts:
        recv_displs.append(offset)
        offset += c

    total_send = sum(send_counts)
    total_recv = sum(recv_counts)

    # Symmetric heap requires all ranks to allocate the same size.
    max_send = torch.tensor([total_send], device=f"cuda:{rank}")
    max_recv = torch.tensor([total_recv], device=f"cuda:{rank}")
    dist.all_reduce(max_send, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_recv, op=dist.ReduceOp.MAX)
    alloc_send = max(int(max_send.item()), 1)
    alloc_recv = max(int(max_recv.item()), 1)

    iris_input = ctx.zeros(alloc_send, dtype=dtype)
    for j in range(world_size):
        if send_counts[j] > 0:
            iris_input[send_displs[j] : send_displs[j] + send_counts[j]] = rank * 100 + j

    iris_output = ctx.zeros(alloc_recv, dtype=dtype)

    # Reference
    ref_input = (
        iris_input[:total_send].clone() if total_send > 0 else torch.zeros(0, dtype=dtype, device=f"cuda:{rank}")
    )
    ref_output = torch.zeros(max(total_recv, 0), dtype=dtype, device=f"cuda:{rank}")

    ctx.barrier()
    if total_send > 0 and total_recv > 0:
        dist.all_to_all_single(
            ref_output[:total_recv],
            ref_input,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
    torch.cuda.synchronize()

    ctx.barrier()
    ctx.ccl.all_to_all_v(
        iris_output,
        iris_input,
        send_counts,
        send_displs,
        recv_counts,
        recv_displs,
    )
    torch.cuda.synchronize()

    atol = 1e-3

    try:
        if total_recv > 0:
            assert torch.allclose(iris_output[:total_recv], ref_output[:total_recv], atol=atol), (
                f"Rank {rank}: max diff = {torch.abs(iris_output[:total_recv] - ref_output[:total_recv]).max().item()}"
            )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()
