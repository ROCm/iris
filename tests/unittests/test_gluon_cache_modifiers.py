# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl
from itertools import product


# === Kernel Definitions ===


@gluon.jit
def load_cache_modifier_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    source_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    cache_modifier: gl.constexpr,
    volatile: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    result = ctx.load(data + offsets, partner, mask=mask, cache_modifier=cache_modifier, volatile=volatile)
    gl.store(results + offsets, result, mask=mask)


@gluon.jit
def store_cache_modifier_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    destination_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    value = gl.load(data + offsets, mask=mask)

    for dst_rank in range(num_ranks):
        ctx.store(results + offsets, value, dst_rank, mask=mask, cache_modifier=cache_modifier)


@gluon.jit
def get_cache_modifier_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    load_cache_modifier: gl.constexpr,
    store_cache_modifier: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    acc = gl.zeros([BLOCK_SIZE], dtype=gl.float32, layout=layout)

    for target_rank in range(num_ranks):
        ctx.get(
            data + offsets,
            results + offsets,
            target_rank,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )
        acc = acc + gl.load(results + offsets, mask=mask)

    gl.store(results + offsets, acc, mask=mask)


@gluon.jit
def put_cache_modifier_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    to_rank: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    load_cache_modifier: gl.constexpr,
    store_cache_modifier: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    ctx.put(
        data + offsets,
        results + offsets,
        to_rank,
        mask=mask,
        load_cache_modifier=load_cache_modifier,
        store_cache_modifier=store_cache_modifier,
    )


@gluon.jit
def copy_local_read_remote_write_cache_modifier_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    load_cache_modifier: gl.constexpr,
    store_cache_modifier: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * cur_rank
        ctx.copy(
            src_data + offsets,
            dest_data + offsets,
            cur_rank,
            target_rank,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


@gluon.jit
def copy_remote_read_local_write_cache_modifier_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    load_cache_modifier: gl.constexpr,
    store_cache_modifier: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    for source_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * source_rank
        dest_data = results + BLOCK_SIZE * source_rank
        ctx.copy(
            src_data + offsets,
            dest_data + offsets,
            source_rank,
            cur_rank,
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
def test_gluon_load_cache_modifiers(cache_modifier, volatile):
    """Test IrisDeviceCtx.load() with various cache modifiers and volatile settings."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    source_rank = ctx.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = ctx.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = (1,)
    load_cache_modifier_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        cache_modifier,
        volatile,
        num_warps=1,
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
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize("cache_modifier", STORE_CACHE_MODIFIERS)
def test_gluon_store_cache_modifiers(cache_modifier):
    """Test IrisDeviceCtx.store() with various cache modifiers."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    destination_rank = ctx.get_rank()

    BLOCK_SIZE = 16
    src = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros_like(src)

    ctx.barrier()

    grid = (1,)
    store_cache_modifier_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        src,
        results,
        destination_rank,
        num_ranks,
        BLOCK_SIZE,
        cache_modifier,
        num_warps=1,
    )
    ctx.barrier()

    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(f"STORE test failed with cache_modifier={cache_modifier}")
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
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_gluon_get_cache_modifiers(load_cache_modifier, store_cache_modifier):
    """Test IrisDeviceCtx.get() with various cache modifiers."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    cur_rank = ctx.get_rank()

    BLOCK_SIZE = 16
    data = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = (1,)
    get_cache_modifier_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        load_cache_modifier,
        store_cache_modifier,
        num_warps=1,
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
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_gluon_put_cache_modifiers_local(load_cache_modifier, store_cache_modifier):
    """Test IrisDeviceCtx.put() local (to_rank == cur_rank) with various cache modifiers."""
    ctx = iris_gl.iris(1 << 20)
    cur_rank = ctx.get_rank()
    context_tensor = ctx.get_device_context()

    BLOCK_SIZE = 16
    data = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros_like(data)

    ctx.barrier()

    grid = (1,)
    put_cache_modifier_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        cur_rank,
        BLOCK_SIZE,
        load_cache_modifier,
        store_cache_modifier,
        num_warps=1,
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
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_gluon_put_cache_modifiers_remote(load_cache_modifier, store_cache_modifier):
    """Test IrisDeviceCtx.put() remote (to_rank != cur_rank) with various cache modifiers."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    cur_rank = ctx.get_rank()
    context_tensor = ctx.get_device_context()

    if num_ranks < 2:
        pytest.skip("Remote put test requires at least 2 ranks")

    BLOCK_SIZE = 16
    data = ctx.ones(BLOCK_SIZE, dtype=torch.float32)
    results = ctx.zeros(BLOCK_SIZE, dtype=torch.float32)

    ctx.barrier()

    remote_rank = (cur_rank + 1) % num_ranks
    grid = (1,)
    if cur_rank == 0:
        put_cache_modifier_kernel[grid](
            iris_gl.IrisDeviceCtx,
            context_tensor,
            data,
            results,
            cur_rank,
            remote_rank,
            BLOCK_SIZE,
            load_cache_modifier,
            store_cache_modifier,
            num_warps=1,
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

    ctx.barrier()
    del ctx
    import gc

    gc.collect()


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_gluon_copy_local_read_remote_write(load_cache_modifier, store_cache_modifier):
    """Test IrisDeviceCtx.copy() local read → remote write with various cache modifiers."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    cur_rank = ctx.get_rank()

    BLOCK_SIZE = 16
    data = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)

    ctx.barrier()

    grid = (1,)
    copy_local_read_remote_write_cache_modifier_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        load_cache_modifier,
        store_cache_modifier,
        num_warps=1,
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

    ctx.barrier()
    del ctx
    import gc

    gc.collect()


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier",
    list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS)),
)
def test_gluon_copy_remote_read_local_write(load_cache_modifier, store_cache_modifier):
    """Test IrisDeviceCtx.copy() remote read → local write with various cache modifiers."""
    ctx = iris_gl.iris(1 << 20)
    num_ranks = ctx.get_num_ranks()
    context_tensor = ctx.get_device_context()
    cur_rank = ctx.get_rank()

    BLOCK_SIZE = 16
    data = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)
    base = cur_rank + num_ranks
    for i in range(num_ranks):
        data[i, :] = base * (i + 1)

    results = ctx.zeros((num_ranks, BLOCK_SIZE), dtype=torch.float32)

    ctx.barrier()

    grid = (1,)
    copy_remote_read_local_write_cache_modifier_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        data,
        results,
        cur_rank,
        num_ranks,
        BLOCK_SIZE,
        load_cache_modifier,
        store_cache_modifier,
        num_warps=1,
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

    ctx.barrier()
    del ctx
    import gc

    gc.collect()
