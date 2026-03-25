# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl
from iris.tracing.events import TraceEvent


@gluon.jit
def device_context_tracing_1d_address_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    dummy_buffer,
    source_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """Test ctx.tracing.record_event_start/end with a 1D address (block of pointers)."""
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=True)

    layout: gl.constexpr = gl.BlockedLayout([1], [BLOCK_SIZE], [1], [0])
    offsets = gl.arange(0, BLOCK_SIZE, layout=layout)
    address_1d = dummy_buffer + offsets

    # All-true mask derived from offsets (offsets are always < BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    handle = ctx.tracing.record_event_start(
        event_id=TraceEvent().put,
        target_rank=(source_rank + 1) % num_ranks,
        address=address_1d,
        pid_m=gl.program_id(0),
        pid_n=0,
        mask=mask,
    )
    ctx.tracing.record_event_end(handle)


def test_device_context_gluon_tracing_1d_address():
    """Test GluonDeviceTracing record_event_start/end with a 1D address block."""
    shmem = iris_gl.iris(1 << 20)
    shmem.tracing.enable(max_events=1000)
    context_tensor = shmem.get_device_context()
    source_rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()

    BLOCK_SIZE = 64  # AMD wavefront size (64 threads per warp)
    # Dummy buffer only to form 1D pointer block; never read/write
    dummy_buffer = shmem.zeros((BLOCK_SIZE,), dtype=torch.int64)

    shmem.barrier()

    device_context_tracing_1d_address_kernel[(1,)](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        dummy_buffer,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        num_warps=1,
    )
    shmem.barrier()

    # Verify we recorded at least one event
    assert shmem.tracing.trace_counter.item() >= 1

    # Verify event data fields for the first recorded event
    bufs = shmem.tracing.trace_buffers
    assert bufs["event_id"][0].item() == 3  # TraceEvent().put == 3
    assert bufs["cur_rank"][0].item() == source_rank
    assert bufs["target_rank"][0].item() == (source_rank + 1) % num_ranks
    assert bufs["timestamp"][0].item() > 0
    # duration_cycles holds the end timestamp; it must be >= start timestamp
    assert bufs["duration_cycles"][0].item() >= bufs["timestamp"][0].item()
    # payload_size: BLOCK_SIZE elements × 8 bytes each (dummy_buffer is int64)
    assert bufs["payload_size"][0].item() == BLOCK_SIZE * 8

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_device_context_gluon_initialize():
    """Test IrisDeviceCtx.initialize() works without tracing enabled."""
    shmem = iris_gl.iris(1 << 20)
    context_tensor = shmem.get_device_context()

    assert context_tensor is not None
    assert isinstance(context_tensor, torch.Tensor)
    assert context_tensor.dtype == torch.int64
    num_ranks = shmem.get_num_ranks()
    # At least [cur_rank, num_ranks, heap_base_0, ...]; layout may add more (e.g. tracing flag)
    assert context_tensor.shape[0] >= 2 + num_ranks
    assert context_tensor[0].item() == shmem.get_rank()
    assert context_tensor[1].item() == num_ranks

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_device_context_gluon_tracing_enable():
    """Test that shmem.tracing.enable() rebuilds context tensor with tracing fields."""
    shmem = iris_gl.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()

    # Without tracing: [cur_rank, num_ranks, heap_base_0..N-1, trace_enabled=0]
    ctx_no_trace = shmem.get_device_context()
    size_no_trace = ctx_no_trace.shape[0]

    # Enable tracing and rebuild
    shmem.tracing.enable(max_events=1000)
    ctx_with_trace = shmem.get_device_context()
    size_with_trace = ctx_with_trace.shape[0]

    # With tracing the context tensor is larger:
    # [cur_rank, num_ranks, heap_base_0..N-1, trace_enabled=1, max_events,
    #  trace_counter_ptr, op_index_counter_ptr, 13 buffer ptrs]
    assert size_with_trace > size_no_trace
    # trace_enabled flag should be 1
    trace_enabled_idx = 2 + num_ranks
    assert ctx_with_trace[trace_enabled_idx].item() == 1

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
