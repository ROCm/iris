# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from iris import DeviceContext


@triton.jit
def device_context_load_kernel(
    context_tensor,
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test DeviceContext.load() method."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    result = ctx.load(data + offsets, from_rank=partner, mask=mask)
    tl.store(results + offsets, result, mask=mask)


@triton.jit
def device_context_store_kernel(
    context_tensor,
    source,
    target,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test DeviceContext.store() method."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    data = tl.load(source + offsets, mask=mask)
    ctx.store(target + offsets, data, to_rank=partner, mask=mask)


@triton.jit
def device_context_atomic_add_kernel(
    context_tensor,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test DeviceContext.atomic_add() method."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    acc = tl.full([BLOCK_SIZE], 1, dtype=results.type.element_ty)

    for target_rank in range(num_ranks):
        ctx.atomic_add(
            results + offsets,
            acc,
            to_rank=target_rank,
            mask=mask,
            sem="acq_rel",
            scope="sys",
        )


@triton.jit
def device_context_atomic_cas_kernel(
    context_tensor,
    flag,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
):
    """Test DeviceContext.atomic_cas() method."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    old = ctx.atomic_cas(flag + pid, 0, 1, to_rank=partner, sem="release", scope="sys")


@triton.jit
def device_context_get_kernel(
    context_tensor,
    source,
    local_buffer,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test DeviceContext.get() method."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    ctx.get(source + offsets, local_buffer + offsets, from_rank=partner, mask=mask)


@triton.jit
def device_context_put_kernel(
    context_tensor,
    local_buffer,
    target,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Test DeviceContext.put() method."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    ctx.put(local_buffer + offsets, target + offsets, to_rank=partner, mask=mask)


# === Test Functions ===


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        1,
        8,
        16,
        32,
    ],
)
def test_device_context_load(dtype, BLOCK_SIZE):
    """Test DeviceContext.load() works correctly."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    rank = ctx.get_rank()
    partner = int((rank + num_ranks // 2) % num_ranks)

    # Get device context tensor
    context_tensor = ctx.get_device_context()

    data = ctx.full((BLOCK_SIZE,), rank, dtype=dtype)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_load_kernel[grid](context_tensor, data, results, rank, num_ranks, BLOCK_SIZE)
    ctx.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda") * partner

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(f"[Rank {rank}] Test failed!")
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        8,
        16,
        32,
    ],
)
def test_device_context_store(dtype, BLOCK_SIZE):
    """Test DeviceContext.store() works correctly."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    context_tensor = ctx.get_device_context()
    source = ctx.full((BLOCK_SIZE,), cur_rank, dtype=dtype)
    target = ctx.zeros_like(source)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_store_kernel[grid](context_tensor, source, target, cur_rank, num_ranks, BLOCK_SIZE)
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda") * partner

    try:
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", target)
        raise
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        8,
        16,
    ],
)
def test_device_context_atomic_add(dtype, BLOCK_SIZE):
    """Test DeviceContext.atomic_add() works correctly."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    context_tensor = ctx.get_device_context()
    results = ctx.zeros(BLOCK_SIZE, dtype=dtype)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_atomic_add_kernel[grid](context_tensor, results, cur_rank, num_ranks, BLOCK_SIZE)
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda") * num_ranks

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
        import gc

        gc.collect()


def test_device_context_atomic_cas():
    """Test DeviceContext.atomic_cas() works correctly."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    context_tensor = ctx.get_device_context()
    flag = ctx.zeros(1, dtype=torch.int32)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_atomic_cas_kernel[grid](context_tensor, flag, cur_rank, num_ranks)
    ctx.barrier()

    expected = torch.tensor([1], dtype=torch.int32, device="cuda")

    try:
        torch.testing.assert_close(flag, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", flag)
        raise
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        8,
        16,
    ],
)
def test_device_context_get(BLOCK_SIZE):
    """Test DeviceContext.get() works correctly."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    context_tensor = ctx.get_device_context()
    source = ctx.full((BLOCK_SIZE,), cur_rank, dtype=torch.float32)
    local_buffer = ctx.zeros((BLOCK_SIZE,), dtype=torch.float32)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_get_kernel[grid](context_tensor, source, local_buffer, cur_rank, num_ranks, BLOCK_SIZE)
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner

    try:
        torch.testing.assert_close(local_buffer, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", local_buffer)
        raise
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        8,
        16,
    ],
)
def test_device_context_put(BLOCK_SIZE):
    """Test DeviceContext.put() works correctly."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    context_tensor = ctx.get_device_context()
    local_buffer = ctx.full((BLOCK_SIZE,), cur_rank, dtype=torch.float32)
    target = ctx.zeros((BLOCK_SIZE,), dtype=torch.float32)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_put_kernel[grid](context_tensor, local_buffer, target, cur_rank, num_ranks, BLOCK_SIZE)
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner

    try:
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", target)
        raise
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


def test_device_context_initialize():
    """Test DeviceContext.initialize() creates valid context."""
    ctx = iris.iris(1 << 20)
    cur_rank = ctx.get_rank()
    num_ranks = ctx.get_num_ranks()

    context_tensor = ctx.get_device_context()

    assert context_tensor is not None
    assert isinstance(context_tensor, torch.Tensor)
    assert context_tensor.dtype == torch.int64
    assert context_tensor.shape[0] == 2 + num_ranks
    assert context_tensor[0].item() == cur_rank
    assert context_tensor[1].item() == num_ranks

    ctx.barrier()
    del ctx
    import gc

    gc.collect()


def test_device_context_imports():
    """Test that DeviceContext is available from correct import paths."""
    from iris import DeviceContext as DC1
    from iris.iris import DeviceContext as DC2

    assert DC1 is DC2
