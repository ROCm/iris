# SPDX-License-Identifier: MIT

"""Compile-and-run tests for the tile / wait-then-put SDMA device ops.

Each exported device op gets a tiny kernel so that a broken @triton.jit body
(which only compiles on first launch) is caught here.
"""

import importlib.util

import pytest
import torch
import triton
import triton.language as tl

import iris

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("xio") is None,
    reason="rocm-xio not installed (pip install iris[sdma])",
)

M, N = 16, 32  # fp32 elements; rows are N * 4 bytes
M_C = tl.constexpr(M)
ROW_BYTES_C = tl.constexpr(N * 4)
ROW_BYTES = N * 4


@triton.jit
def _put_tile_signal_kernel(src, dst, flag, from_rank: tl.constexpr, to_rank: tl.constexpr, heap_bases, ctx):
    iris.put_tile_signal(
        src, dst, from_rank, to_rank, heap_bases, ctx, flag, 1, ROW_BYTES_C, M_C, ROW_BYTES_C, ROW_BYTES_C
    )
    iris.quiet(ctx, to_rank)


@triton.jit
def _wait_then_put_tile_kernel(src, dst, go, from_rank: tl.constexpr, to_rank: tl.constexpr, heap_bases, ctx):
    iris.wait_then_put_tile(
        src, dst, from_rank, to_rank, heap_bases, ctx, go, 1, ROW_BYTES_C, M_C, ROW_BYTES_C, ROW_BYTES_C
    )
    # The SDMA queue is blocked on `go`; release it from the device.
    tl.atomic_add(go, 1)
    iris.quiet(ctx, to_rank)


@triton.jit
def _wait_then_put_tile_signal_kernel(
    src, dst, go, flag, from_rank: tl.constexpr, to_rank: tl.constexpr, heap_bases, ctx
):
    iris.wait_then_put_tile_signal(
        src, dst, from_rank, to_rank, heap_bases, ctx, go, 1, flag, 1, ROW_BYTES_C, M_C, ROW_BYTES_C, ROW_BYTES_C
    )
    tl.atomic_add(go, 1)
    iris.quiet(ctx, to_rank)


@triton.jit
def _wait_then_put_tile_batch_kernel(
    src,
    dst,
    go,
    row_offsets,
    col_offsets,
    widths,
    heights,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    ctx,
):
    iris.wait_then_put_tile_batch(
        src, dst, from_rank, to_rank, heap_bases, ctx, go, 1, row_offsets, col_offsets, widths, heights,
        0, 2, 4, ROW_BYTES_C, ROW_BYTES_C, 2,
    )  # fmt: skip
    tl.atomic_add(go, 1)
    iris.quiet(ctx, to_rank)


@triton.jit
def _wait_then_put_tile_batch_signal_kernel(
    src,
    dst,
    go,
    flag,
    row_offsets,
    col_offsets,
    widths,
    heights,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    ctx,
):
    iris.wait_then_put_tile_batch_signal(
        src, dst, from_rank, to_rank, heap_bases, ctx, go, 1, flag, 1, row_offsets, col_offsets, widths, heights,
        0, 2, 4, ROW_BYTES_C, ROW_BYTES_C, 2,
    )  # fmt: skip
    tl.atomic_add(go, 1)
    iris.quiet(ctx, to_rank)


def _setup():
    shmem = iris.iris(1 << 22, engines=("sdma",))
    if shmem.get_num_ranks() != 2:
        pytest.skip("Copy engine tests require exactly two ranks.")
    rank = shmem.get_rank()
    src = shmem.zeros(M * N, device="cuda", dtype=torch.float32)
    src.copy_(torch.arange(M * N, dtype=torch.float32, device=src.device))
    dst = shmem.zeros(M * N, device="cuda", dtype=torch.float32)
    go = shmem.zeros((1,), device="cuda", dtype=torch.int32)
    flag = shmem.zeros((1,), device="cuda", dtype=torch.int32)
    return shmem, rank, 1 - rank, src, dst, go, flag


def _args(shmem):
    return shmem.get_heap_bases(), shmem.get_copy_engine_ctx()


def _tile_metadata(device):
    # Two row blocks (rows 0-1 and 2-3 of a 4-row region), full row width.
    row_offsets = torch.tensor([0, 2], dtype=torch.int32, device=device)
    col_offsets = torch.zeros(2, dtype=torch.int32, device=device)
    widths = torch.full((2,), ROW_BYTES, dtype=torch.int32, device=device)
    heights = torch.full((2,), 2, dtype=torch.int32, device=device)
    return row_offsets, col_offsets, widths, heights


def test_put_tile_signal():
    shmem, rank, peer, src, dst, _, flag = _setup()
    if rank == 0:
        _put_tile_signal_kernel[(1,)](src, dst, flag, rank, peer, *_args(shmem))
    shmem.barrier()
    if rank == 1:
        assert torch.equal(dst, src)
        assert flag.item() == 1
    shmem.barrier()
    del shmem


def test_wait_then_put_tile():
    shmem, rank, peer, src, dst, go, _ = _setup()
    if rank == 0:
        _wait_then_put_tile_kernel[(1,)](src, dst, go, rank, peer, *_args(shmem))
    shmem.barrier()
    if rank == 1:
        assert torch.equal(dst, src)
    shmem.barrier()
    del shmem


def test_wait_then_put_tile_signal():
    shmem, rank, peer, src, dst, go, flag = _setup()
    if rank == 0:
        _wait_then_put_tile_signal_kernel[(1,)](src, dst, go, flag, rank, peer, *_args(shmem))
    shmem.barrier()
    if rank == 1:
        assert torch.equal(dst, src)
        assert flag.item() == 1
    shmem.barrier()
    del shmem


def test_wait_then_put_tile_batch():
    shmem, rank, peer, src, dst, go, _ = _setup()
    meta = _tile_metadata(src.device)
    if rank == 0:
        _wait_then_put_tile_batch_kernel[(1,)](src, dst, go, *meta, rank, peer, *_args(shmem))
    shmem.barrier()
    if rank == 1:
        assert torch.equal(dst[: 4 * N], src[: 4 * N])
    shmem.barrier()
    del shmem


def test_wait_then_put_tile_batch_signal():
    shmem, rank, peer, src, dst, go, flag = _setup()
    meta = _tile_metadata(src.device)
    if rank == 0:
        _wait_then_put_tile_batch_signal_kernel[(1,)](src, dst, go, flag, *meta, rank, peer, *_args(shmem))
    shmem.barrier()
    if rank == 1:
        assert torch.equal(dst[: 4 * N], src[: 4 * N])
        assert flag.item() == 1
    shmem.barrier()
    del shmem


# ============================================================================
# Edge cases: dtype sizes, partial/offset tile lists, wait semantics
# ============================================================================


@triton.jit
def _put_dtype_kernel(src, dst, num_elements, from_rank: tl.constexpr, to_rank: tl.constexpr, heap_bases, ctx):
    offsets = tl.arange(0, 128)
    mask = offsets < num_elements
    iris.put(
        src + offsets,
        dst + offsets,
        from_rank,
        to_rank,
        heap_bases,
        mask=mask,
        copy_engine_ctx=ctx,
        use_copy_engine=True,
        contiguous_copy=True,
    )
    iris.quiet(ctx, to_rank)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.float16, torch.bfloat16, torch.float64])
def test_put_element_size_from_dtype(dtype):
    """The copy size must follow the pointee dtype (1/2/8 bytes), not a 4-byte default."""
    shmem = iris.iris(1 << 20, engines=("sdma",))
    if shmem.get_num_ranks() != 2:
        pytest.skip("Copy engine tests require exactly two ranks.")
    rank = shmem.get_rank()
    n = 100
    src = shmem.zeros(128, device="cuda", dtype=dtype)
    src.copy_(torch.arange(128, dtype=torch.float32, device=src.device).to(dtype))
    dst = shmem.zeros(128, device="cuda", dtype=dtype)
    if rank == 0:
        _put_dtype_kernel[(1,)](src, dst, n, rank, 1, shmem.get_heap_bases(), shmem.get_copy_engine_ctx())
    shmem.barrier()
    if rank == 1:
        assert torch.equal(dst[:n], src[:n])
        assert not dst[n:].any(), "copy ran past the masked region"
    shmem.barrier()
    del shmem


@triton.jit
def _partial_tile_batch_kernel(
    src,
    dst,
    go,
    row_offsets,
    col_offsets,
    widths,
    heights,
    transfer_start,
    transfer_count,
    from_rank: tl.constexpr,
    to_rank: tl.constexpr,
    heap_bases,
    ctx,
):
    iris.wait_then_put_tile_batch(
        src, dst, from_rank, to_rank, heap_bases, ctx, go, 1, row_offsets, col_offsets, widths, heights,
        transfer_start, transfer_count, 4, ROW_BYTES_C, ROW_BYTES_C, 4,
    )  # fmt: skip
    tl.atomic_add(go, 1)
    iris.quiet(ctx, to_rank)


def test_wait_then_put_tile_batch_partial_with_offsets():
    """transfer_count < MAX_TILES, non-zero transfer_start and column offsets: only the selected tiles are copied."""
    shmem, rank, peer, src, dst, go, _ = _setup()
    dev = src.device
    # Entry 0 must be skipped (transfer_start=1). Entries 1 and 2 are copied.
    row_offsets = torch.tensor([12, 4, 10], dtype=torch.int32, device=dev)
    col_offsets = torch.tensor([0, 8, 0], dtype=torch.int32, device=dev)
    widths = torch.tensor([ROW_BYTES, 8 * 4, ROW_BYTES], dtype=torch.int32, device=dev)
    heights = torch.tensor([2, 4, 2], dtype=torch.int32, device=dev)
    if rank == 0:
        _partial_tile_batch_kernel[(1,)](
            src, dst, go, row_offsets, col_offsets, widths, heights, 1, 2, rank, peer, *_args(shmem)
        )
    shmem.barrier()
    if rank == 1:
        s, d = src.view(M, N), dst.view(M, N)
        expected = torch.zeros_like(s)
        expected[4:8, 8:16] = s[4:8, 8:16]
        expected[10:12, :] = s[10:12, :]
        assert torch.equal(d, expected)
    shmem.barrier()
    del shmem


@triton.jit
def _post_wait_then_put_tile_kernel(src, dst, go, from_rank: tl.constexpr, to_rank: tl.constexpr, heap_bases, ctx):
    iris.wait_then_put_tile(
        src, dst, from_rank, to_rank, heap_bases, ctx, go, 1, ROW_BYTES_C, M_C, ROW_BYTES_C, ROW_BYTES_C
    )


def test_wait_then_put_tile_blocks_until_flag():
    """The copy must not happen before the wait flag is released."""
    import time

    shmem, rank, peer, src, dst, go, _ = _setup()
    if rank == 0:
        _post_wait_then_put_tile_kernel[(1,)](src, dst, go, rank, peer, *_args(shmem))
    shmem.barrier()
    if rank == 1:
        time.sleep(0.5)
        assert not dst.any(), "SDMA copy ran before the wait flag was released"
    shmem.barrier()
    if rank == 0:
        go.fill_(1)
        torch.cuda.synchronize()
    shmem.barrier()
    if rank == 1:
        deadline = time.time() + 10
        while time.time() < deadline and not torch.equal(dst, src):
            time.sleep(0.01)
        assert torch.equal(dst, src), "SDMA copy did not complete after the flag was released"
    shmem.barrier()
    del shmem
