# SPDX-License-Identifier: MIT

import pytest
import torch
import triton
import triton.language as tl

import iris


@triton.jit
def _copy_engine_linear_kernel(
    src,
    dst,
    num_elements,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    copy_engine_ctx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    iris.put(
        src + offsets,
        dst + offsets,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx,
        mask=mask,
        USE_COPY_ENGINE=True,
    )


def _require_two_ranks(shmem):
    if shmem.get_num_ranks() != 2:
        pytest.skip("Copy engine tests require exactly two ranks.")


def _make_expected(size, dtype, device):
    return torch.arange(size, dtype=dtype, device=device)


def _allocate_symmetric_range(shmem, size, dtype):
    # Each rank gets an identical symmetric tensor with deterministic values.
    values = torch.arange(size, dtype=dtype, device=shmem.get_device())
    tensor = shmem.zeros(size, device="cuda", dtype=dtype)
    tensor.copy_(values)
    return tensor


def _make_grid(n, block):
    return lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)


@pytest.mark.parametrize("num_elements", [256, 1024])
def test_copy_engine_device_linear_put(num_elements):
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    src = _allocate_symmetric_range(shmem, num_elements, torch.float32)
    dst = shmem.zeros(num_elements, device="cuda", dtype=torch.float32)

    grid = _make_grid(num_elements, 128)
    if rank == 0:
        _copy_engine_linear_kernel[grid](
            src,
            dst,
            num_elements,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            BLOCK_SIZE=128,
        )

    shmem.barrier()

    if rank == 1:
        expected = _make_expected(num_elements, torch.float32, dst.device)
        assert torch.allclose(dst, expected)

    shmem.barrier()
    del shmem


@pytest.mark.parametrize("num_elements", [512, 2048])
def test_copy_engine_host_put(num_elements):
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    src = _allocate_symmetric_range(shmem, num_elements, torch.float32)
    dst = shmem.zeros(num_elements, device="cuda", dtype=torch.float32)

    if rank == 0:
        shmem.put(src, dst_rank=remote_rank, dst_tensor=dst, async_op=True)
        shmem.quiet(dst_rank=remote_rank)

    shmem.barrier()

    if rank == 1:
        expected = _make_expected(num_elements, torch.float32, dst.device)
        assert torch.allclose(dst, expected)

    shmem.barrier()
    del shmem


@triton.jit
def _copy_engine_atomic_kernel(
    flag,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    copy_engine_ctx,
    increment: tl.constexpr,
):
    iris.atomic_add(
        flag,
        increment,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx=copy_engine_ctx,
        USE_COPY_ENGINE=True,
    )


def test_copy_engine_atomic_add():
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    flag = shmem.zeros((1,), device="cuda", dtype=torch.int32)

    if rank == 0:
        _copy_engine_atomic_kernel[(1,)](
            flag,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            increment=5,
        )

    shmem.barrier()

    if rank == 1:
        assert flag.item() == 5

    shmem.barrier()
    del shmem


# ============================================================================
# Copy Engine Atomic CAS Tests
# ============================================================================


@triton.jit
def _copy_engine_atomic_cas_kernel(
    flag,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    copy_engine_ctx,
    compare: tl.constexpr,
    value: tl.constexpr,
):
    iris.atomic_cas(
        flag,
        compare,
        value,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx=copy_engine_ctx,
        USE_COPY_ENGINE=True,
    )


def test_copy_engine_atomic_cas():
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    flag = shmem.zeros((1,), device="cuda", dtype=torch.int32)

    if rank == 0:
        _copy_engine_atomic_cas_kernel[(1,)](
            flag,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            compare=0,
            value=1,
        )

    shmem.barrier()

    if rank == 1:
        assert flag.item() == 1

    shmem.barrier()
    del shmem


# ============================================================================
# 2D/Tiled Copy Tests
# ============================================================================


@triton.jit
def _copy_engine_2d_kernel(
    src_base,
    dst_base,
    num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    src_stride: tl.constexpr,
    dst_stride: tl.constexpr,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    copy_engine_ctx,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D tiled copy using strided parameters."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create 2D pointer blocks
    src_ptrs = src_base + offs_m[:, None] * src_stride + offs_n[None, :]
    dst_ptrs = dst_base + offs_m[:, None] * dst_stride + offs_n[None, :]

    # Create mask
    mask = (offs_m[:, None] < num_rows) & (offs_n[None, :] < num_cols)

    # 2D copy with strides
    iris.put(
        src_ptrs,
        dst_ptrs,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx,
        stride_fm=src_stride,
        stride_tm=dst_stride,
        mask=mask,
        USE_COPY_ENGINE=True,
        IS_2D_COPY=True,
        from_base_ptr=src_base,
        to_base_ptr=dst_base,
    )


@pytest.mark.parametrize("M,N", [(16, 16), (32, 64)])
def test_copy_engine_2d_tiled(M, N):
    """Test 2D tiled copy with strides."""
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    # Allocate 2D matrices with row-major layout
    stride = N  # Elements per row
    src = _allocate_symmetric_range(shmem, M * N, torch.float32).view(M, N)
    dst = shmem.zeros(M * N, device="cuda", dtype=torch.float32).view(M, N)

    BLOCK_M, BLOCK_N = 8, 16
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    if rank == 0:
        _copy_engine_2d_kernel[(grid_m, grid_n)](
            src,
            dst,
            M,
            N,
            stride,
            stride,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    shmem.barrier()

    if rank == 1:
        expected = _make_expected(M * N, torch.float32, dst.device).view(M, N)
        assert torch.allclose(dst, expected)

    shmem.barrier()
    del shmem


# ============================================================================
# Combined Operations Tests (put + signal, wait + put)
# ============================================================================


@triton.jit
def _copy_engine_put_signal_kernel(
    src,
    dst,
    flag,
    num_elements,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    copy_engine_ctx,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy data and signal completion with atomic add."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Copy data
    iris.put(
        src + offsets,
        dst + offsets,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx,
        mask=mask,
        USE_COPY_ENGINE=True,
    )

    # Signal completion (last thread in block)
    if pid == 0:
        iris.atomic_add(
            flag,
            1,
            from_rank,
            to_rank,
            heap_bases,
            copy_engine_ctx=copy_engine_ctx,
            USE_COPY_ENGINE=True,
        )


def test_copy_engine_put_with_signal():
    """Test copy followed by atomic signal."""
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    num_elements = 512
    src = _allocate_symmetric_range(shmem, num_elements, torch.float32)
    dst = shmem.zeros(num_elements, device="cuda", dtype=torch.float32)
    flag = shmem.zeros((1,), device="cuda", dtype=torch.int32)

    grid = _make_grid(num_elements, 128)
    if rank == 0:
        _copy_engine_put_signal_kernel[grid](
            src,
            dst,
            flag,
            num_elements,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            BLOCK_SIZE=128,
        )

    shmem.barrier()

    if rank == 1:
        # Check data transferred
        expected = _make_expected(num_elements, torch.float32, dst.device)
        assert torch.allclose(dst, expected)
        # Check signal received
        assert flag.item() == 1

    shmem.barrier()
    del shmem


# ============================================================================
# Multi-Block Concurrent Operations
# ============================================================================


@triton.jit
def _copy_engine_multi_block_kernel(
    src,
    dst,
    counters,
    num_elements,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    copy_engine_ctx,
    BLOCK_SIZE: tl.constexpr,
):
    """Multiple blocks concurrently using copy engine."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Each block copies its chunk
    iris.put(
        src + offsets,
        dst + offsets,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx,
        mask=mask,
        USE_COPY_ENGINE=True,
    )

    # Each block atomically increments its counter
    iris.atomic_add(
        counters + pid,
        1,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx=copy_engine_ctx,
        USE_COPY_ENGINE=True,
    )


@pytest.mark.parametrize("num_blocks", [4, 8])
def test_copy_engine_multi_block_concurrent(num_blocks):
    """Test multiple workgroups using copy engine concurrently."""
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    BLOCK_SIZE = 128
    num_elements = num_blocks * BLOCK_SIZE
    src = _allocate_symmetric_range(shmem, num_elements, torch.float32)
    dst = shmem.zeros(num_elements, device="cuda", dtype=torch.float32)
    counters = shmem.zeros(num_blocks, device="cuda", dtype=torch.int32)

    if rank == 0:
        _copy_engine_multi_block_kernel[(num_blocks,)](
            src,
            dst,
            counters,
            num_elements,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    shmem.barrier()

    if rank == 1:
        # Check all data transferred
        expected = _make_expected(num_elements, torch.float32, dst.device)
        assert torch.allclose(dst, expected)
        # Check all blocks signaled
        assert torch.all(counters == 1).item()

    shmem.barrier()
    del shmem


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


@pytest.mark.skip(reason="Zero-size transfers may cause SDMA queue issues - needs investigation")
def test_copy_engine_zero_size():
    """Test copy engine with zero-size transfer (should be no-op)."""
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    src = shmem.zeros(128, device="cuda", dtype=torch.float32)
    dst = shmem.zeros(128, device="cuda", dtype=torch.float32)

    if rank == 0:
        # Empty slice should be a no-op
        shmem.put(src[:0], dst_rank=remote_rank, dst_tensor=dst[:0], async_op=True)
        shmem.quiet(dst_rank=remote_rank)

    shmem.barrier()

    # Destination should still be zeros
    if rank == 1:
        assert torch.all(dst == 0).item()

    shmem.barrier()
    del shmem


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_copy_engine_different_dtypes(dtype):
    """Test copy engine with different data types."""
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    num_elements = 256
    src = _allocate_symmetric_range(shmem, num_elements, dtype)
    dst = shmem.zeros(num_elements, device="cuda", dtype=dtype)

    grid = _make_grid(num_elements, 128)
    if rank == 0:
        _copy_engine_linear_kernel[grid](
            src,
            dst,
            num_elements,
            rank,
            remote_rank,
            shmem.get_heap_bases(),
            shmem.get_copy_engine_ctx(),
            BLOCK_SIZE=128,
        )

    shmem.barrier()

    if rank == 1:
        expected = _make_expected(num_elements, dtype, dst.device)
        assert torch.allclose(dst, expected)

    shmem.barrier()
    del shmem


def test_copy_engine_bidirectional():
    """Test both ranks doing copy engine operations simultaneously."""
    shmem = iris.iris(1 << 20)
    _require_two_ranks(shmem)

    rank = shmem.get_rank()
    remote_rank = 1 - rank

    num_elements = 256
    # Each rank has its own data to send
    src = _allocate_symmetric_range(shmem, num_elements, torch.float32)
    # Scale by rank to make data different
    src.mul_(rank + 1)
    dst = shmem.zeros(num_elements, device="cuda", dtype=torch.float32)

    grid = _make_grid(num_elements, 128)
    # Both ranks send their data
    _copy_engine_linear_kernel[grid](
        src,
        dst,
        num_elements,
        rank,
        remote_rank,
        shmem.get_heap_bases(),
        shmem.get_copy_engine_ctx(),
        BLOCK_SIZE=128,
    )

    shmem.barrier()

    # Each rank should have received the other's data
    expected = _make_expected(num_elements, torch.float32, dst.device) * (remote_rank + 1)
    assert torch.allclose(dst, expected)

    shmem.barrier()
    del shmem
