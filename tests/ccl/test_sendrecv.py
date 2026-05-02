# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for point-to-point send/recv operations.

Ring test: rank i sends to rank (i+1) % world_size and receives from
rank (i-1) % world_size. Verifies data integrity by filling each rank's
send buffer with a deterministic value and checking the received value
matches the sender's rank.
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
    "M, N",
    [
        (128, 64),  # Small
        (1024, 256),  # Medium
        (8192, 8192),  # Large
    ],
)
def test_sendrecv_ring(dtype, M, N):
    """
    Ring send/recv: each rank sends to next and receives from previous.

    Verifies that the received data matches the sender's rank value.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size < 2:
        del shmem
        gc.collect()
        pytest.skip("sendrecv requires at least 2 ranks")

    dst = (rank + 1) % world_size
    src = (rank - 1) % world_size

    # Send buffer: filled with this rank's value
    send_buf = shmem.zeros((M, N), dtype=dtype)
    send_buf.fill_(float(rank + 1))

    # Recv buffer: will be filled by sender
    recv_buf = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Use sendrecv for simultaneous send + recv
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.sendrecv(send_buf, recv_buf, dst=dst, src=src, config=config)
    torch.cuda.synchronize()

    # Expected: recv_buf should contain src rank's value
    expected_value = float(src + 1)
    expected = torch.full((M, N), expected_value, dtype=dtype, device=f"cuda:{rank}")

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(recv_buf - expected).max().item()

    try:
        assert torch.allclose(recv_buf, expected, atol=atol), (
            f"Rank {rank}: max diff {max_diff}, expected value {expected_value} from rank {src}"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 64),  # Small
        (1024, 256),  # Medium
    ],
)
def test_send_recv_paired(dtype, M, N):
    """
    Explicit send/recv pair: rank 0 sends to rank 1, rank 1 receives.

    Other ranks are idle. Tests the basic send + recv API without sendrecv.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size < 2:
        del shmem
        gc.collect()
        pytest.skip("send/recv requires at least 2 ranks")

    tensor = shmem.zeros((M, N), dtype=dtype)
    config = Config(block_size_m=32, block_size_n=64)

    shmem.barrier()

    if rank == 0:
        tensor.fill_(42.0)
        shmem.ccl.send(tensor, dst=1, config=config)
    elif rank == 1:
        shmem.ccl.recv(tensor, src=0, config=config)
    else:
        # send/recv use device_barrier internally — all ranks must participate
        shmem.device_barrier()

    torch.cuda.synchronize()

    if rank == 1:
        expected = torch.full((M, N), 42.0, dtype=dtype, device=f"cuda:{rank}")
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(tensor - expected).max().item()
        assert torch.allclose(tensor, expected, atol=atol), f"Rank 1: max diff {max_diff}, expected 42.0 from rank 0"

    try:
        pass
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float32])
def test_sendrecv_single_element(dtype):
    """Edge case: send/recv a single element tensor."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size < 2:
        del shmem
        gc.collect()
        pytest.skip("sendrecv requires at least 2 ranks")

    dst = (rank + 1) % world_size
    src = (rank - 1) % world_size

    send_buf = shmem.zeros((1, 1), dtype=dtype)
    send_buf.fill_(float(rank + 100))

    recv_buf = shmem.zeros((1, 1), dtype=dtype)

    shmem.barrier()

    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.sendrecv(send_buf, recv_buf, dst=dst, src=src, config=config)
    torch.cuda.synchronize()

    expected_value = float(src + 100)
    expected = torch.full((1, 1), expected_value, dtype=dtype, device=f"cuda:{rank}")

    try:
        assert torch.allclose(recv_buf, expected, atol=1e-5), (
            f"Rank {rank}: expected {expected_value}, got {recv_buf.item()}"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
