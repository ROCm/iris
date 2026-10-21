# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-host jobs must fail fast instead of falling back to the single-host FD-passing mesh."""

import itertools
import socket
import threading
import time
import types

import pytest
import torch.distributed as dist

import iris.host.distributed.topology as topology_module
from iris.host.distributed import fd_passing
from iris.host.distributed.topology import FabricInfo
from iris.host.memory import symmetric_heap
from iris.host.memory.symmetric_heap import SymmetricHeap

POD = "3e568d58317a4bff84fcb661a751878f:2"


class _Constructed(Exception):
    """Raised by a faked allocator: the guard let the heap reach allocator construction."""


def _constructed(*args, **kwargs):
    raise _Constructed()


def _fake_topology(hosts_and_domains):
    gpu_info = {
        rank: types.SimpleNamespace(
            hostname=host, fabric_info=FabricInfo(*domain.split(":")) if domain else FabricInfo()
        )
        for rank, (host, domain) in enumerate(hosts_and_domains)
    }
    return types.SimpleNamespace(gpu_info=gpu_info)


@pytest.fixture
def heap_env(monkeypatch):
    """Fake everything SymmetricHeap.__init__ touches before the allocator exists."""
    monkeypatch.delenv("IRIS_ALLOCATOR", raising=False)
    monkeypatch.setattr(symmetric_heap, "is_simulation_env", lambda: False)
    monkeypatch.setattr(topology_module, "_fabric_failure", None)
    for name in ("TorchAllocator", "VMemAllocator", "VMemChunkedAllocator"):
        monkeypatch.setattr(symmetric_heap, name, _constructed)

    def use(hostnames, topology=None, discover_error=None):
        monkeypatch.setattr(symmetric_heap, "_hostnames_by_rank", lambda num_ranks: list(hostnames))

        class FakeDiscovery:
            def discover(self):
                if discover_error is not None:
                    raise discover_error
                return topology

        monkeypatch.setattr(topology_module, "TopologyDiscovery", FakeDiscovery)

    return use


def _heap(allocator_type, num_ranks=4):
    return SymmetricHeap(1 << 20, 0, 0, num_ranks, allocator_type)


class TestSingleHostAllocators:
    @pytest.mark.parametrize("allocator_type", ["torch", "vmem"])
    def test_multi_host_rejected_before_allocator(self, heap_env, allocator_type):
        heap_env(["tray-0", "tray-0", "tray-1", "tray-1"])

        with pytest.raises(RuntimeError, match="IRIS_ALLOCATOR=vmem_chunked") as err:
            _heap(allocator_type)
        assert f"IRIS_ALLOCATOR='{allocator_type}'" in str(err.value)
        assert "tray-0, tray-1" in str(err.value)

    @pytest.mark.parametrize("allocator_type", ["torch", "vmem"])
    def test_single_host_allowed(self, heap_env, allocator_type):
        heap_env(["tray-0"] * 4)

        with pytest.raises(_Constructed):
            _heap(allocator_type)

    def test_single_rank_skips_hostname_exchange(self, monkeypatch, heap_env):
        def no_exchange(num_ranks):
            raise AssertionError("hostname exchange should not run for one rank")

        monkeypatch.setattr(symmetric_heap, "_hostnames_by_rank", no_exchange)

        with pytest.raises(_Constructed):
            _heap("vmem", num_ranks=1)


class TestVMemChunkedMultiHost:
    def test_shared_fabric_domain_allowed(self, heap_env):
        hosts = [("tray-0", POD), ("tray-0", POD), ("tray-1", POD), ("tray-1", POD)]
        heap_env([h for h, _ in hosts], _fake_topology(hosts))

        with pytest.raises(_Constructed):
            _heap("vmem_chunked")

    def test_missing_fabric_domain_rejected(self, monkeypatch, heap_env):
        hosts = [("tray-0", ""), ("tray-0", ""), ("tray-1", ""), ("tray-1", "")]
        heap_env([h for h, _ in hosts], _fake_topology(hosts))
        monkeypatch.setattr(topology_module, "_fabric_failure", "amdsmi_init() failed with status 8")

        with pytest.raises(RuntimeError, match="one fabric domain") as err:
            _heap("vmem_chunked")
        message = str(err.value)
        assert "tray-0: ranks [0, 1], fabric domain <none>" in message
        assert "tray-1: ranks [2, 3], fabric domain <none>" in message
        assert "Rank 0 found no fabric domain: amdsmi_init() failed with status 8." in message
        assert "amd-smi fabric -i" in message

    def test_different_pods_rejected(self, heap_env):
        other = "aa" * 16 + ":5"
        hosts = [("tray-0", POD), ("tray-1", other)]
        heap_env([h for h, _ in hosts], _fake_topology(hosts))

        with pytest.raises(RuntimeError, match="one fabric domain") as err:
            _heap("vmem_chunked", num_ranks=2)
        assert f"tray-1: ranks [1], fabric domain {other}" in str(err.value)

    def test_discovery_failure_is_fatal_on_multi_host(self, heap_env):
        heap_env(["tray-0", "tray-1"], discover_error=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="needs topology discovery") as err:
            _heap("vmem_chunked", num_ranks=2)
        assert isinstance(err.value.__cause__, RuntimeError)

    def test_discovery_failure_still_warns_on_single_host(self, heap_env):
        heap_env(["tray-0", "tray-0"], discover_error=RuntimeError("boom"))

        with pytest.raises(_Constructed):
            _heap("vmem_chunked", num_ranks=2)


def test_hostnames_exchanged_through_store(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.data = {}

        def set(self, key, value):
            self.data[key] = value.encode("utf-8")

        def get(self, key):
            return self.data[key]

    store = FakeStore()
    monkeypatch.setattr(dist.distributed_c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(symmetric_heap.socket, "gethostname", lambda: "tray-1")
    monkeypatch.setattr(symmetric_heap, "_hostname_exchanges", itertools.count(1))
    store.data["iris_hostname_v1/0"] = b"tray-0"

    assert symmetric_heap._hostnames_by_rank(2) == ["tray-0", "tray-1"]


def test_fd_mesh_accept_times_out_on_missing_peer(monkeypatch, tmp_path):
    monkeypatch.setattr(fd_passing, "_ACCEPT_TIMEOUT_S", 0.2)
    paths = {0: str(tmp_path / "r0.sock"), 1: str(tmp_path / "r1.sock")}

    with pytest.raises(TimeoutError, match=r"ranks \[1\] to connect to rank 0"):
        fd_passing.setup_fd_mesh(0, 2, paths)
    assert not (tmp_path / "r0.sock").exists()


def test_fd_mesh_handshake_times_out_on_silent_peer(monkeypatch, tmp_path):
    monkeypatch.setattr(fd_passing, "_HANDSHAKE_TIMEOUT_S", 0.2)
    paths = {0: str(tmp_path / "r0.sock"), 1: str(tmp_path / "r1.sock")}
    silent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def connect_without_sending_rank():
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                silent.connect(paths[0])
                return
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.01)

    peer = threading.Thread(target=connect_without_sending_rank)
    peer.start()
    try:
        with pytest.raises(TimeoutError, match="did not send its rank"):
            fd_passing.setup_fd_mesh(0, 2, paths)
    finally:
        peer.join()
        silent.close()
    assert not (tmp_path / "r0.sock").exists()


def test_vmem_allocator_rejects_more_ranks_than_local_gpus(monkeypatch):
    from iris.host.memory.allocators import vmem_allocator

    monkeypatch.setattr(vmem_allocator.torch.cuda, "device_count", lambda: 4)

    with pytest.raises(RuntimeError, match="world_size=8 and this process sees 4 GPUs"):
        vmem_allocator.VMemAllocator(1 << 21, 0, 0, 8)
