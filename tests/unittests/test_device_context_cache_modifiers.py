# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from iris import DeviceContext
from itertools import product


# === Kernel Definitions ===


@triton.jit
def device_context_load_cache_modifier_kernel(
    context_tensor,
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    cache_modifier: tl.constexpr,
    volatile: tl.constexpr,
):
    """Test DeviceContext.load() with cache_modifier and volatile."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    result = ctx.load(
        data + offsets,
        from_rank=partner,
        mask=mask,
        cache_modifier=cache_modifier,
        volatile=volatile,
    )
    tl.store(results + offsets, result, mask=mask)


@triton.jit
def device_context_store_cache_modifier_kernel(
    context_tensor,
    source,
    target,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    to_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Test DeviceContext.store() with cache_modifier."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    data = tl.load(source + offsets, mask=mask)
    ctx.store(target + offsets, data, to_rank=to_rank, mask=mask, cache_modifier=cache_modifier)


@triton.jit
def device_context_get_cache_modifier_kernel(
    context_tensor,
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    """Test DeviceContext.get() with load_cache_modifier and store_cache_modifier."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    acc = tl.zeros([BLOCK_SIZE], dtype=data.type.element_ty)

    for target_rank in range(num_ranks):
        ctx.get(
            data + offsets,
            results + offsets,
            from_rank=target_rank,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )
        acc += tl.load(results + offsets, mask=mask)

    tl.store(results + offsets, acc, mask=mask)


@triton.jit
def device_context_put_cache_modifier_kernel(
    context_tensor,
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    to_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    """Test DeviceContext.put() with load_cache_modifier and store_cache_modifier."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    ctx.put(
        data + offsets,
        results + offsets,
        to_rank=to_rank,
        mask=mask,
        load_cache_modifier=load_cache_modifier,
        store_cache_modifier=store_cache_modifier,
    )


@triton.jit
def device_context_copy_local_read_remote_write_kernel(
    context_tensor,
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    """Test DeviceContext.copy() with cache modifiers (local read, remote write)."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * cur_rank
        ctx.copy(
            src_data + offsets,
            dest_data + offsets,
            from_rank=cur_rank,
            to_rank=target_rank,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


@triton.jit
def device_context_copy_remote_read_local_write_kernel(
    context_tensor,
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    """Test DeviceContext.copy() with cache modifiers (remote read, local write)."""
    ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for source_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * source_rank
        dest_data = results + BLOCK_SIZE * source_rank
        ctx.copy(
            src_data + offsets,
            dest_data + offsets,
            from_rank=source_rank,
            to_rank=cur_rank,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


# === Cache modifier lists ===

LOAD_CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
STORE_CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]
VOLATILE_OPTIONS = [False, True]


# === Test Functions ===


@pytest.mark.parametrize("cache_modifier,volatile", list(product(LOAD_CACHE_MODIFIERS, VOLATILE_OPTIONS)))
def test_device_context_load_cache_modifiers(cache_modifier, volatile):
    """Test DeviceContext.load() with various cache modifiers and volatile settings."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()
    partner = int((cur_rank + num_ranks // 2) % num_ranks)

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.full((BLOCK_SIZE,), cur_rank, dtype=torch.float32)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_load_cache_modifier_kernel[grid](
        context_tensor, data, results, cur_rank, num_ranks, BLOCK_SIZE, cache_modifier, volatile
    )
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(f"LOAD test failed with cache_modifier={cache_modifier}, volatile={volatile}")
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise


@pytest.mark.parametrize("cache_modifier", STORE_CACHE_MODIFIERS)
def test_device_context_store_cache_modifiers_local(cache_modifier):
    """Test DeviceContext.store() local (from_rank == to_rank) with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    # For local store, we need partner == cur_rank; use a different kernel approach.
    # We'll test with partner = cur_rank by calling the kernel but verifying store to self.
    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    source = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    target = ctx.zeros(BLOCK_SIZE, dtype=torch.float32)

    ctx.barrier()

    # We override the kernel to store to itself (to_rank == cur_rank).
    @triton.jit
    def local_store_kernel(
        context_tensor,
        source,
        target,
        cur_rank: tl.constexpr,
        num_ranks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        cache_modifier: tl.constexpr,
    ):
        ctx = DeviceContext.initialize(context_tensor, cur_rank, num_ranks)
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < BLOCK_SIZE
        data = tl.load(source + offsets, mask=mask)
        ctx.store(target + offsets, data, to_rank=cur_rank, mask=mask, cache_modifier=cache_modifier)

    grid = lambda meta: (1,)
    local_store_kernel[grid](context_tensor, source, target, cur_rank, num_ranks, BLOCK_SIZE, cache_modifier)
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
    try:
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(f"LOCAL STORE test failed with cache_modifier={cache_modifier}")
        print(e)
        raise


@pytest.mark.parametrize("cache_modifier", STORE_CACHE_MODIFIERS)
def test_device_context_store_cache_modifiers_remote(cache_modifier):
    """Test DeviceContext.store() remote (from_rank != to_rank) with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    if num_ranks < 2:
        pytest.skip("Remote store test requires at least 2 ranks")

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    source = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    target = ctx.zeros(BLOCK_SIZE, dtype=torch.float32)

    ctx.barrier()

    remote_rank = (cur_rank + 1) % num_ranks
    grid = lambda meta: (1,)
    if cur_rank == 0:
        device_context_store_cache_modifier_kernel[grid](
            context_tensor, source, target, cur_rank, num_ranks, remote_rank, BLOCK_SIZE, cache_modifier
        )

    ctx.barrier()

    if cur_rank == 1:
        expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
        try:
            torch.testing.assert_close(target, expected, rtol=0, atol=0)
        except AssertionError as e:
            print(f"REMOTE STORE test failed with cache_modifier={cache_modifier}")
            print(e)
            raise


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_device_context_get_cache_modifiers(load_cache_modifier, store_cache_modifier):
    """Test DeviceContext.get() with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_get_cache_modifier_kernel[grid](
        context_tensor, data, results, cur_rank, num_ranks, BLOCK_SIZE, load_cache_modifier, store_cache_modifier
    )
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * num_ranks

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(
            f"GET test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_device_context_put_cache_modifiers_local(load_cache_modifier, store_cache_modifier):
    """Test DeviceContext.put() local (from_rank == to_rank) with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_put_cache_modifier_kernel[grid](
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        cur_rank,
        BLOCK_SIZE,
        load_cache_modifier,
        store_cache_modifier,
    )
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(
            f"LOCAL PUT test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
        print(e)
        raise


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_device_context_put_cache_modifiers_remote(load_cache_modifier, store_cache_modifier):
    """Test DeviceContext.put() remote (from_rank != to_rank) with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    if num_ranks < 2:
        pytest.skip("Remote put test requires at least 2 ranks")

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros(BLOCK_SIZE, dtype=torch.float32)

    ctx.barrier()

    remote_rank = (cur_rank + 1) % num_ranks
    grid = lambda meta: (1,)
    if cur_rank == 0:
        device_context_put_cache_modifier_kernel[grid](
            context_tensor,
            data,
            results,
            cur_rank,
            num_ranks,
            remote_rank,
            BLOCK_SIZE,
            load_cache_modifier,
            store_cache_modifier,
        )

    ctx.barrier()

    if cur_rank == 1:
        expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")
        try:
            torch.testing.assert_close(results, expected, rtol=0, atol=0)
        except AssertionError as e:
            print(
                f"REMOTE PUT test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
            )
            print(e)
            raise


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_device_context_copy_local_read_remote_write(load_cache_modifier, store_cache_modifier):
    """Test DeviceContext.copy() local read → remote write with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_copy_local_read_remote_write_kernel[grid](
        context_tensor, data, results, cur_rank, num_ranks, BLOCK_SIZE, load_cache_modifier, store_cache_modifier
    )

    ctx.barrier()

    for rank_id in range(num_ranks):
        expected_value = (rank_id + num_ranks) * (rank_id + 1)
        assert torch.allclose(
            results[rank_id],
            torch.full((BLOCK_SIZE,), expected_value, dtype=torch.float32, device=results.device),
        ), (
            f"Mismatch at rank {cur_rank}, slot {rank_id} with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier",
    list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS)),
)
def test_device_context_copy_remote_read_local_write(load_cache_modifier, store_cache_modifier):
    """Test DeviceContext.copy() remote read → local write with various cache modifiers."""
    ctx = iris.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()

    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)

    ctx.barrier()

    grid = lambda meta: (1,)
    device_context_copy_remote_read_local_write_kernel[grid](
        context_tensor, data, results, cur_rank, num_ranks, BLOCK_SIZE, load_cache_modifier, store_cache_modifier
    )

    ctx.barrier()

    for rank_id in range(num_ranks):
        expected_value = (rank_id + num_ranks) * (rank_id + 1)
        assert torch.allclose(
            results[rank_id],
            torch.full((BLOCK_SIZE,), expected_value, dtype=torch.float32, device=results.device),
        ), (
            f"Mismatch at rank {cur_rank}, slot {rank_id} with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
