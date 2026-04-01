# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for as_remote() API — both host-side (Iris, IrisGluon) and device-side (IrisDeviceCtx).
"""

import gc

import torch
import pytest
import iris
import iris.experimental.iris_gluon as iris_gl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------
# Host-side tests (single-process, no torchrun required)
# ---------------------------------------------------------------------------


class TestAsRemoteHostIris:
    """Host-side as_remote() tests using the Iris (Triton) backend."""

    def test_host_basic(self):
        """as_remote returns a tensor with matching shape/dtype/strides but different data_ptr."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        num_ranks = ctx.num_ranks
        if num_ranks < 2:
            pytest.skip("Need >= 2 ranks")

        buf = ctx.zeros(128, dtype=torch.float32)
        target = (ctx.cur_rank + 1) % num_ranks
        remote = ctx.as_remote(buf, target)

        assert remote.shape == buf.shape
        assert remote.dtype == buf.dtype
        assert remote.stride() == buf.stride()
        assert remote.data_ptr() != buf.data_ptr()

    def test_host_pointer_math(self):
        """Offset from respective heap base must be identical."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        num_ranks = ctx.num_ranks
        if num_ranks < 2:
            pytest.skip("Need >= 2 ranks")

        buf = ctx.zeros(64, dtype=torch.float32)
        target = (ctx.cur_rank + 1) % num_ranks

        local_base = int(ctx.heap.heap_bases[ctx.cur_rank].item())
        remote_base = int(ctx.heap.heap_bases[target].item())

        remote = ctx.as_remote(buf, target)
        assert remote.data_ptr() - remote_base == buf.data_ptr() - local_base

    def test_host_self_rank(self):
        """as_remote(tensor, cur_rank) returns a tensor with the same data_ptr."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        buf = ctx.zeros(64, dtype=torch.float32)
        remote = ctx.as_remote(buf, ctx.cur_rank)
        assert remote.data_ptr() == buf.data_ptr()
        assert remote.shape == buf.shape

    def test_host_non_symmetric_raises(self):
        """as_remote on a non-symmetric tensor raises ValueError."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        external = torch.zeros(64, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="symmetric heap"):
            ctx.as_remote(external, 0)

    def test_host_rank_out_of_range(self):
        """as_remote with invalid rank raises ValueError."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        buf = ctx.zeros(64, dtype=torch.float32)
        with pytest.raises(ValueError, match="out of range"):
            ctx.as_remote(buf, ctx.num_ranks)
        with pytest.raises(ValueError, match="out of range"):
            ctx.as_remote(buf, -1)

    def test_host_non_contiguous(self):
        """as_remote preserves strides of a non-contiguous (sliced) tensor."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        buf_2d = ctx.zeros(16, 16, dtype=torch.float32)
        sliced = buf_2d[::2, ::2]  # non-contiguous view
        assert not sliced.is_contiguous()

        remote = ctx.as_remote(sliced, ctx.cur_rank)
        assert remote.shape == sliced.shape
        assert remote.stride() == sliced.stride()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_host_multi_dtype(self, dtype):
        """as_remote works across multiple dtypes."""
        ctx = iris.iris(1 << 20, allocator_type="torch")
        buf = ctx.zeros(64, dtype=dtype)
        remote = ctx.as_remote(buf, ctx.cur_rank)
        assert remote.dtype == dtype
        assert remote.shape == buf.shape


# ---------------------------------------------------------------------------
# Host-side tests for IrisGluon backend (multi-GPU, needs torchrun)
# ---------------------------------------------------------------------------


class TestAsRemoteHostGluon:
    """Host-side as_remote() tests using the IrisGluon (Gluon) backend."""

    def test_host_basic(self):
        """as_remote returns a tensor with matching shape/dtype/strides but different data_ptr."""
        ctx = iris_gl.iris(1 << 20)
        num_ranks = ctx.get_num_ranks()
        cur_rank = ctx.get_rank()
        if num_ranks < 2:
            ctx.barrier()
            del ctx
            gc.collect()
            pytest.skip("Need >= 2 ranks")

        buf = ctx.zeros(128, dtype=torch.float32)
        target = (cur_rank + 1) % num_ranks
        remote = ctx.as_remote(buf, target)

        assert remote.shape == buf.shape
        assert remote.dtype == buf.dtype
        assert remote.stride() == buf.stride()
        assert remote.data_ptr() != buf.data_ptr()

        ctx.barrier()
        del ctx
        gc.collect()

    def test_host_pointer_math(self):
        """Offset from respective heap base must be identical."""
        ctx = iris_gl.iris(1 << 20)
        num_ranks = ctx.get_num_ranks()
        cur_rank = ctx.get_rank()
        if num_ranks < 2:
            ctx.barrier()
            del ctx
            gc.collect()
            pytest.skip("Need >= 2 ranks")

        buf = ctx.zeros(64, dtype=torch.float32)
        target = (cur_rank + 1) % num_ranks

        local_base = int(ctx.heap.heap_bases[cur_rank].item())
        remote_base = int(ctx.heap.heap_bases[target].item())

        remote = ctx.as_remote(buf, target)
        assert remote.data_ptr() - remote_base == buf.data_ptr() - local_base

        ctx.barrier()
        del ctx
        gc.collect()

    def test_host_self_rank(self):
        """as_remote(tensor, cur_rank) returns a tensor with the same data_ptr."""
        ctx = iris_gl.iris(1 << 20)
        cur_rank = ctx.get_rank()

        buf = ctx.zeros(64, dtype=torch.float32)
        remote = ctx.as_remote(buf, cur_rank)
        assert remote.data_ptr() == buf.data_ptr()
        assert remote.shape == buf.shape

        ctx.barrier()
        del ctx
        gc.collect()

    def test_host_non_symmetric_raises(self):
        """as_remote on a non-symmetric tensor raises ValueError."""
        ctx = iris_gl.iris(1 << 20)
        external = torch.zeros(64, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="symmetric heap"):
            ctx.as_remote(external, 0)

        ctx.barrier()
        del ctx
        gc.collect()

    def test_host_rank_out_of_range(self):
        """as_remote with invalid rank raises ValueError."""
        ctx = iris_gl.iris(1 << 20)
        buf = ctx.zeros(64, dtype=torch.float32)
        with pytest.raises(ValueError, match="out of range"):
            ctx.as_remote(buf, ctx.get_num_ranks())
        with pytest.raises(ValueError, match="out of range"):
            ctx.as_remote(buf, -1)

        ctx.barrier()
        del ctx
        gc.collect()

    def test_host_non_contiguous(self):
        """as_remote preserves strides of a non-contiguous (sliced) tensor."""
        ctx = iris_gl.iris(1 << 20)
        buf_2d = ctx.zeros(16, 16, dtype=torch.float32)
        sliced = buf_2d[::2, ::2]
        assert not sliced.is_contiguous()

        remote = ctx.as_remote(sliced, ctx.get_rank())
        assert remote.shape == sliced.shape
        assert remote.stride() == sliced.stride()

        ctx.barrier()
        del ctx
        gc.collect()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_host_multi_dtype(self, dtype):
        """as_remote works across multiple dtypes."""
        ctx = iris_gl.iris(1 << 20)
        buf = ctx.zeros(64, dtype=dtype)
        remote = ctx.as_remote(buf, ctx.get_rank())
        assert remote.dtype == dtype
        assert remote.shape == buf.shape

        ctx.barrier()
        del ctx
        gc.collect()


# ---------------------------------------------------------------------------
# Device-side tests (multi-GPU, needs torchrun)
# ---------------------------------------------------------------------------


@gluon.jit
def as_remote_read_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    source_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """Read from a remote rank using ctx.as_remote + gl.load."""
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    # Translate pointer then load directly (instead of ctx.load)
    remote_ptr = ctx.as_remote(data + offsets, partner)
    result = gl.load(remote_ptr, mask=mask)
    gl.store(results + offsets, result, mask=mask)


@gluon.jit
def as_remote_write_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    destination_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """Write to a remote rank using ctx.as_remote + gl.store."""
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    value = gl.load(data + offsets, mask=mask)

    # Translate pointer then store directly (instead of ctx.store)
    for dst_rank in range(num_ranks):
        remote_ptr = ctx.as_remote(results + offsets, dst_rank)
        gl.store(remote_ptr, value, mask=mask)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
@pytest.mark.parametrize("BLOCK_SIZE", [16, 32])
def test_device_as_remote_read(dtype, BLOCK_SIZE):
    """Rank reads from its partner using ctx.as_remote + gl.load."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    source_rank = ctx.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    data = ctx.full((BLOCK_SIZE,), source_rank, dtype=dtype)
    results = ctx.zeros_like(data)

    ctx.barrier()

    as_remote_read_kernel[(1,)](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda") * partner

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
@pytest.mark.parametrize("BLOCK_SIZE", [16, 32])
def test_device_as_remote_write(dtype, BLOCK_SIZE):
    """Each rank writes 1s to all ranks using ctx.as_remote + gl.store."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    destination_rank = ctx.get_rank()

    src = ctx.ones(BLOCK_SIZE, dtype=dtype)
    results = ctx.zeros_like(src)

    ctx.barrier()

    as_remote_write_kernel[(1,)](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        src,
        results,
        destination_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
    finally:
        ctx.barrier()
        del ctx
        gc.collect()
