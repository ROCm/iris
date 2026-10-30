# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import importlib.util
import os
import socket
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris

BLOCK_SIZE = 64
HAS_XIO = importlib.util.find_spec("xio") is not None


@triton.jit
def put_to_every_rank(data, results, cur_rank: tl.constexpr, num_ranks: tl.constexpr, BLOCK: tl.constexpr, heap_bases):
    offsets = tl.arange(0, BLOCK)
    for target_rank in range(num_ranks):
        iris.put(data + offsets, results + cur_rank * BLOCK + offsets, cur_rank, target_rank, heap_bases)


@triton.jit
def copy_engine_put(
    src, dst, from_rank: tl.constexpr, to_rank: tl.constexpr, BLOCK: tl.constexpr, heap_bases, copy_engine_ctx
):
    offsets = tl.arange(0, BLOCK)
    iris.put(
        src + offsets,
        dst + offsets,
        from_rank,
        to_rank,
        heap_bases,
        copy_engine_ctx=copy_engine_ctx,
        use_copy_engine=True,
        contiguous_copy=True,
    )
    iris.quiet(copy_engine_ctx, to_rank)


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


def _check_copy_engine(shmem, rank, num_ranks):
    peer = (rank + 1) % num_ranks
    src = shmem.full((BLOCK_SIZE,), rank + 1, dtype=torch.int32)
    host_dst = shmem.zeros(num_ranks * BLOCK_SIZE, dtype=torch.int32)
    device_dst = shmem.zeros(num_ranks * BLOCK_SIZE, dtype=torch.int32)
    shmem.barrier()
    shmem.put(src, to_rank=peer, to_tensor=host_dst[rank * BLOCK_SIZE : (rank + 1) * BLOCK_SIZE])
    copy_engine_put[(1,)](
        src,
        device_dst[rank * BLOCK_SIZE :],
        rank,
        peer,
        BLOCK_SIZE,
        shmem.get_heap_bases(),
        shmem.get_copy_engine_ctx(),
    )
    torch.cuda.synchronize()
    shmem.barrier()
    sender = (rank - 1) % num_ranks
    expected = torch.full((BLOCK_SIZE,), sender + 1, dtype=torch.int32, device=src.device)
    for dst in (host_dst, device_dst):
        torch.testing.assert_close(dst[sender * BLOCK_SIZE : (sender + 1) * BLOCK_SIZE], expected, rtol=0, atol=0)


def _placed_rank(rank, bus_ids, gpus, allocator_type, engines, reverse_visibility, port):
    # The spawned world must not join the runner's torchrun store or inherit its rank variables.
    for name in ("TORCHELASTIC_USE_AGENT_STORE", "IRIS_ALLOCATOR", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(name, None)
    visible = list(range(len(bus_ids)))
    if reverse_visibility and rank % 2:
        # This rank numbers the GPUs in the opposite order, so device ordinals differ between the ranks.
        # Set before anything initializes HIP in this process, which reads it once.
        parent = os.environ.get("HIP_VISIBLE_DEVICES")
        parent = parent.split(",") if parent else [str(i) for i in visible]
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join(reversed(parent))
        visible.reverse()
    ordinal = {physical: local for local, physical in enumerate(visible)}
    gpu = ordinal[gpus[rank]]
    assert torch.cuda.get_device_properties(gpu).pci_bus_id == bus_ids[gpus[rank]]
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=len(gpus),
        device_id=torch.device("cuda", gpu),
        timeout=timedelta(minutes=3),
    )
    try:
        shmem = iris.iris(1 << 26, allocator_type=allocator_type, engines=engines, device=gpu)
        assert shmem.gpu_id == gpu
        assert torch.cuda.current_device() == gpu
        assert shmem.get_heap_bases().device == torch.device("cuda", gpu)
        assert shmem.get_device_context().device == torch.device("cuda", gpu)

        if allocator_type in ("vmem", "vmem_chunked"):
            # A missing grant page-faults the GPU, so check access before anything touches the heap.
            granted = {ordinal[g] for g in gpus} if allocator_type == "vmem" else {gpu}
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

        if engines:
            _check_copy_engine(shmem, rank, num_ranks)

        assert all(torch.cuda.memory_allocated(i) == 0 for i in range(torch.cuda.device_count()) if i != gpu)
        shmem.barrier()
        del shmem
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "allocator_type, engines, reverse_visibility",
    [
        ("torch", (), False),
        ("vmem", (), False),
        ("vmem_chunked", (), False),
        ("vmem", (), True),
        pytest.param("torch", ("sdma",), False, marks=pytest.mark.skipif(not HAS_XIO, reason="needs rocm-xio")),
        pytest.param("vmem", ("sdma",), False, marks=pytest.mark.skipif(not HAS_XIO, reason="needs rocm-xio")),
        pytest.param("torch", ("sdma",), True, marks=pytest.mark.skipif(not HAS_XIO, reason="needs rocm-xio")),
        pytest.param("vmem", ("sdma",), True, marks=pytest.mark.skipif(not HAS_XIO, reason="needs rocm-xio")),
    ],
)
def test_ranks_placed_off_rank_mod_gpus(allocator_type, engines, reverse_visibility):
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
            bus_ids = [torch.cuda.get_device_properties(i).pci_bus_id for i in range(num_gpus)]
            mp.spawn(
                _placed_rank,
                args=(bus_ids, gpus, allocator_type, engines, reverse_visibility, port),
                nprocs=len(gpus),
                join=True,
            )
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


def test_device_accepts_numpy_integer():
    gpu = torch.cuda.current_device()
    shmem = iris.iris(1 << 20, device=np.int64(gpu))
    assert shmem.gpu_id == gpu and type(shmem.gpu_id) is int
    del shmem


@pytest.mark.parametrize("device", [-1, 1 << 10, "cpu", torch.device("cpu")])
def test_device_rejects_invalid(device):
    with pytest.raises(ValueError):
        iris.iris(1 << 20, device=device)


def test_device_rejects_bool():
    with pytest.raises(TypeError):
        iris.iris(1 << 20, device=True)


def test_device_must_match_process_group():
    """The runner binds the process group to the current device; any other GPU is rejected on every rank."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip("needs 2 GPUs")
    other = (torch.cuda.current_device() + 1) % num_gpus
    with pytest.raises(ValueError, match="process group is bound to"):
        iris.iris(1 << 20, device=other)
    assert torch.cuda.current_device() != other


@pytest.mark.skipif(not HAS_XIO, reason="needs rocm-xio")
def test_vmem_reuses_copy_engine_gpu_exchange(monkeypatch):
    """With vmem and SDMA, ranks exchange their GPUs once, not once per feature."""
    from iris.host.memory.allocators import vmem_allocator

    def exchange_again(device_id, world_size):
        raise AssertionError("VMemAllocator exchanged rank GPUs that Iris already had")

    monkeypatch.setattr(vmem_allocator, "rank_gpu_ordinals", exchange_again)
    shmem = iris.iris(1 << 20, allocator_type="vmem", engines="sdma")
    assert shmem.heap.allocator is not None
    del shmem


def test_sdma_rejects_multi_host_on_every_rank(monkeypatch):
    """Every rank raises, so no rank waits in a collective for one that already failed."""
    import iris.device.sdma_utils as sdma_utils
    from iris.host.distributed import helpers

    if dist.get_world_size() < 2:
        pytest.skip("needs 2 ranks")
    monkeypatch.setattr(sdma_utils, "bind_sdma_ep", lambda: object())
    monkeypatch.setattr(helpers.socket, "gethostname", lambda: f"node{dist.get_rank()}")
    with pytest.raises(ValueError, match="does not reach GPUs on other hosts"):
        iris.iris(1 << 20, engines="sdma")
