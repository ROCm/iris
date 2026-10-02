# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# SDMA packet emitters, checked by writing packets into an ordinary buffer (not a queue).

import pytest
import torch
import triton
import triton.language as tl
from xio import sdma_ep

import iris
from iris.device import sdma_utils

SENTINEL = 0x5A5A5A5A
ADDR = 0x00007F12_34567890


def _buffer(n_dw=32):
    return torch.full((n_dw,), SENTINEL, dtype=torch.int32, device="cuda")


def _dwords(buf, n):
    return [v & 0xFFFFFFFF for v in buf[:n].cpu().tolist()]


def _nop(n_dw):
    return [(n_dw - 1) << 16] + [0] * (n_dw - 1)


@triton.jit
def _emit_copy(buf, addr, size):
    sdma_utils.place_copy_packet(buf, tl.full((), 0, tl.uint64), size.to(tl.uint32), addr, addr)


@triton.jit
def _emit_sub_window(buf, addr, width, height):
    sdma_utils.place_sub_window_copy_packet(
        buf, tl.full((), 0, tl.uint64), addr, addr, width.to(tl.uint32), height.to(tl.uint32), 64, 64, 0, 0, 0, 0
    )


@triton.jit
def _emit_atomic(buf, addr, src, cmp, OP: tl.constexpr, RETURN: tl.constexpr, IS_64_BIT: tl.constexpr):
    sdma_utils.place_atomic_packet(buf, tl.full((), 0, tl.uint64), addr, src, cmp, OP, RETURN, IS_64_BIT)


def test_copy_packet():
    buf = _buffer()
    _emit_copy[(1,)](buf, ADDR, 4096)
    assert _dwords(buf, 2) == [1, 4095]


def test_zero_size_copy_emits_nop():
    buf = _buffer()
    _emit_copy[(1,)](buf, ADDR, 0)
    assert _dwords(buf, 8) == _nop(7) + [SENTINEL]


@pytest.mark.parametrize("width, height", [(0, 4), (64, 0)])
def test_empty_sub_window_copy_emits_nop(width, height):
    buf = _buffer()
    _emit_sub_window[(1,)](buf, ADDR, width, height)
    assert _dwords(buf, 21) == _nop(20) + [SENTINEL]


# Linux amdgpu vega10_enum.h TC_OP: ATOMIC_{ADD,CMPSWAP}_{RTN_,}{32,64}
@pytest.mark.parametrize(
    "op, ret, is_64_bit, expected",
    [
        (15, True, False, 0x0F),
        (15, True, True, 0x2F),
        (15, False, False, 0x4F),
        (15, False, True, 0x6F),
        (8, True, False, 0x08),
        (8, True, True, 0x28),
        (8, False, False, 0x48),
        (8, False, True, 0x68),
    ],
)
def test_atomic_opcode(op, ret, is_64_bit, expected):
    buf = _buffer()
    src, cmp = 0x00000002_00000003, 0x00000004_00000005
    _emit_atomic[(1,)](buf, ADDR, src, cmp, OP=op, RETURN=ret, IS_64_BIT=is_64_bit)
    hi = (lambda x: x >> 32) if is_64_bit else (lambda x: 0)
    assert _dwords(buf, 7) == [(expected << 25) | 0xA, ADDR & 0xFFFFFFFF, ADDR >> 32, 3, hi(src), 5, hi(cmp)]


# put() through a copy-engine context whose queue, pointers and doorbell are ordinary buffers.
def _fake_copy_engine_ctx():
    queue = torch.full((sdma_ep.SDMA_QUEUE_SIZE // 4,), SENTINEL, dtype=torch.int32, device="cuda")
    ptrs = torch.zeros(5, dtype=torch.int64, device="cuda")  # rptr, wptr, doorbell, cached wptr, committed wptr
    ctx = torch.zeros(sdma_ep.QUEUE_DEVICE_CTX_SIZE, dtype=torch.int64, device="cuda")
    ctx[0] = queue.data_ptr()
    for i in range(5):
        ctx[1 + i] = ptrs.data_ptr() + 8 * i
    return ctx, queue, ptrs


@triton.jit
def _put_1d_kernel(src, dst, n, heap_bases, ctx, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    iris.put(
        src + offsets,
        dst + offsets,
        0,
        0,
        heap_bases,
        mask=offsets < n,
        copy_engine_ctx=ctx,
        use_copy_engine=True,
        contiguous_copy=True,
    )


@triton.jit
def _put_2d_kernel(src, dst, n, heap_bases, ctx, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    iris.put(
        src + rows * STRIDE + cols,
        dst + rows * STRIDE + cols,
        0,
        0,
        heap_bases,
        mask=(rows < n) & (cols < n),
        copy_engine_ctx=ctx,
        from_row_stride=STRIDE,
        to_row_stride=STRIDE,
        use_copy_engine=True,
        contiguous_copy=True,
        from_base_ptr=src,
        to_base_ptr=dst,
    )


@pytest.mark.parametrize("n", [0, 5])
def test_put_1d_masked(n):
    ctx, queue, ptrs = _fake_copy_engine_ctx()
    src = torch.zeros(64, dtype=torch.float32, device="cuda")
    dst = torch.zeros(64, dtype=torch.float32, device="cuda")
    heap_bases = torch.zeros(1, dtype=torch.int64, device="cuda")
    _put_1d_kernel[(1,)](src, dst, n, heap_bases, ctx, BLOCK=64)
    n_dw = sdma_ep.COPY_LINEAR_COMMAND_BYTES // 4
    if n == 0:
        assert _dwords(queue, n_dw + 1) == _nop(n_dw) + [SENTINEL]
    else:
        assert _dwords(queue, 2) == [1, n * 4 - 1]
    # wptr and doorbell advance past the packet
    assert ptrs[1:3].tolist() == [n_dw * 4, n_dw * 4]


@pytest.mark.parametrize("n", [0, 5])
def test_put_2d_masked(n):
    ctx, queue, ptrs = _fake_copy_engine_ctx()
    src = torch.zeros(64 * 64, dtype=torch.float32, device="cuda")
    dst = torch.zeros(64 * 64, dtype=torch.float32, device="cuda")
    heap_bases = torch.zeros(1, dtype=torch.int64, device="cuda")
    _put_2d_kernel[(1,)](src, dst, n, heap_bases, ctx, STRIDE=64, BLOCK=16)
    n_dw = sdma_ep.COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES // 4
    if n == 0:
        assert _dwords(queue, n_dw + 1) == _nop(n_dw) + [SENTINEL]
    else:
        # DW 17-18: rect width (bytes) and height (rows), 1-based
        assert _dwords(queue, 19)[17:] == [n * 4 - 1, n - 1]
    assert ptrs[1:3].tolist() == [n_dw * 4, n_dw * 4]
