# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris

BLOCK_SIZE = 64


@triton.jit
def put_to_every_rank(data, results, cur_rank: tl.constexpr, num_ranks: tl.constexpr, BLOCK: tl.constexpr, heap_bases):
    offsets = tl.arange(0, BLOCK)
    for target_rank in range(num_ranks):
        iris.put(data + offsets, results + cur_rank * BLOCK + offsets, cur_rank, target_rank, heap_bases)


def _access_flags(ptr, device_id):
    from iris.host.platform import hip

    location = hip.hipMemLocation()
    location.type = hip.hipMemLocationTypeDevice
    location.id = device_id
    flags = ctypes.c_ulonglong(0)
    get_access = hip.gpu_runtime.hipMemGetAccess
    get_access.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(hip.hipMemLocation), ctypes.c_void_p]
    get_access.restype = ctypes.c_int
    assert get_access(ctypes.byref(flags), ctypes.byref(location), ctypes.c_void_p(ptr)) == 0
    return flags.value


def _placed_rank(rank, gpus, allocator_type, port):
    gpu = gpus[rank]
    # The spawned world must not join the runner's torchrun store or inherit its rank variables.
    for name in ("TORCHELASTIC_USE_AGENT_STORE", "IRIS_ALLOCATOR", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(name, None)
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=len(gpus),
        device_id=torch.device("cuda", gpu),
        timeout=timedelta(minutes=3),
    )
    try:
        shmem = iris.iris(1 << 26, allocator_type=allocator_type, device=gpu)
        assert shmem.gpu_id == gpu
        assert torch.cuda.current_device() == gpu
        assert shmem.get_heap_bases().device == torch.device("cuda", gpu)
        assert shmem.get_device_context().device == torch.device("cuda", gpu)

        if allocator_type in ("vmem", "vmem_chunked"):
            # A missing grant page-faults the GPU, so check access before anything touches the heap.
            granted = gpus if allocator_type == "vmem" else (gpu,)
            allocator = shmem.heap.allocator
            for device_id in range(torch.cuda.device_count()):
                want = 3 if device_id in granted else 0
                assert _access_flags(allocator.base_va, device_id) == want, f"GPU {device_id} access"

        num_ranks = shmem.get_num_ranks()
        results = shmem.zeros(num_ranks * BLOCK_SIZE, dtype=torch.int32)
        data = shmem.full((BLOCK_SIZE,), rank + 1, dtype=torch.int32)
        shmem.barrier()
        put_to_every_rank[(1,)](data, results, rank, num_ranks, BLOCK_SIZE, shmem.get_heap_bases())
        shmem.barrier()
        expected = torch.arange(1, num_ranks + 1, dtype=torch.int32, device=results.device)
        torch.testing.assert_close(results, expected.repeat_interleave(BLOCK_SIZE), rtol=0, atol=0)

        assert all(torch.cuda.memory_allocated(i) == 0 for i in range(torch.cuda.device_count()) if i != gpu)
        shmem.barrier()
        del shmem
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("allocator_type", ["torch", "vmem", "vmem_chunked"])
def test_ranks_placed_off_rank_mod_gpus(allocator_type):
    """Ranks whose process group is bound to a GPU other than ``rank % visible GPUs``."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 3:
        pytest.skip("needs 3 GPUs so the placement differs from rank % visible GPUs")
    try:
        if dist.get_rank() == 0:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            gpus = (num_gpus - 2, num_gpus - 1)
            mp.spawn(_placed_rank, args=(gpus, allocator_type, port), nprocs=len(gpus), join=True)
    finally:
        dist.barrier()


def test_device_defaults_to_rank_mod_gpus():
    shmem = iris.iris(1 << 20)
    assert shmem.gpu_id == shmem.get_rank() % torch.cuda.device_count()
    del shmem


def test_device_cuda_uses_current_device():
    shmem = iris.iris(1 << 20, device="cuda")
    assert shmem.gpu_id == torch.cuda.current_device()
    del shmem


@pytest.mark.parametrize("device", [-1, 1 << 10, "cpu", torch.device("cpu")])
def test_device_rejects_invalid(device):
    with pytest.raises(ValueError):
        iris.iris(1 << 20, device=device)


def test_sdma_rejects_non_default_placement(monkeypatch):
    import iris.device.sdma_utils as sdma_utils

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip("needs 2 GPUs")
    monkeypatch.setattr(sdma_utils, "bind_sdma_ep", lambda: None)
    other_gpu = (dist.get_rank() + 1) % num_gpus
    with pytest.raises(ValueError, match="sdma"):
        iris.iris(1 << 20, engines="sdma", device=other_gpu)
