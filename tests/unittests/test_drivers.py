# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the driver layer.

These are pure unit tests and should run without GPUs or a distributed setup.
"""

from __future__ import annotations

import pytest
import torch

from iris.drivers.base import (
    DriverError,
    DriverNotSupported,
    ExportableMemory,
    LocalAllocation,
    PeerMapping,
)
from iris.drivers.fabric import fabric_handle_bytes_to_tensor, fabric_tensor_to_handle_bytes
from iris.drivers.fabric import amd as amd_driver_module
from iris.drivers.fabric import nvidia as nvidia_driver_module
from iris.drivers.fabric.amd import (
    AMD_FABRIC_HANDLE_BYTES,
    AmdFabricDriver,
    AmdFabricError,
    AmdFabricNotSupported,
    _amdsmi_bdf_from_pci_bus_id,
    _normalize_fabric_handle_bytes as _normalize_amd_fabric_handle_bytes,
    _round_up as _amd_round_up,
    _valid_fabric_domain,
)
from iris.drivers.fabric.nvidia import (
    CudaFabricError,
    CudaFabricNotSupported,
    FABRIC_HANDLE_BYTES,
    NvidiaFabricDriver,
    _normalize_fabric_handle_bytes,
    _round_up,
)
from iris.host.distributed.topology import InterconnectLevel


class TestExceptions:
    def test_cuda_fabric_error_uses_driver_error_hierarchy(self):
        assert issubclass(CudaFabricError, DriverError)
        assert issubclass(CudaFabricNotSupported, DriverNotSupported)

    def test_amd_fabric_error_uses_driver_error_hierarchy(self):
        assert issubclass(AmdFabricError, DriverError)
        assert issubclass(AmdFabricNotSupported, DriverNotSupported)


class TestFabricHandleSerialization:
    def test_round_trip(self):
        original = bytes(range(FABRIC_HANDLE_BYTES))
        tensor = fabric_handle_bytes_to_tensor(original, "cpu", expected_num_bytes=FABRIC_HANDLE_BYTES)
        assert tensor.shape == (FABRIC_HANDLE_BYTES,)
        assert tensor.dtype == torch.uint8
        assert fabric_tensor_to_handle_bytes(tensor, expected_num_bytes=FABRIC_HANDLE_BYTES) == original

    def test_wrong_size_bytes(self):
        with pytest.raises(ValueError, match="64 bytes"):
            fabric_handle_bytes_to_tensor(b"\x00" * 32, "cpu", expected_num_bytes=FABRIC_HANDLE_BYTES)

    def test_wrong_size_tensor(self):
        with pytest.raises(ValueError, match="64 elements"):
            fabric_tensor_to_handle_bytes(torch.zeros(32, dtype=torch.uint8), expected_num_bytes=FABRIC_HANDLE_BYTES)

    def test_wrong_tensor_dtype(self):
        with pytest.raises(ValueError, match="dtype torch.uint8"):
            fabric_tensor_to_handle_bytes(torch.zeros(FABRIC_HANDLE_BYTES, dtype=torch.int32))


class TestNvidiaFabricHelpers:
    def test_round_up(self):
        assert _round_up(1, 64) == 64
        assert _round_up(64, 64) == 64
        assert _round_up(65, 64) == 128

    def test_round_up_rejects_nonpositive_granularity(self):
        with pytest.raises(ValueError, match="granularity must be > 0"):
            _round_up(1, 0)

    def test_normalize_fabric_handle_bytes(self):
        original = bytes(range(FABRIC_HANDLE_BYTES))
        assert _normalize_fabric_handle_bytes(original) == original
        assert _normalize_fabric_handle_bytes(bytearray(original)) == original
        assert _normalize_fabric_handle_bytes(memoryview(original)) == original
        tensor = torch.tensor(list(original), dtype=torch.uint8)
        assert _normalize_fabric_handle_bytes(tensor) == original

    def test_normalize_fabric_handle_bytes_rejects_wrong_size(self):
        with pytest.raises(CudaFabricError, match="expected 64 bytes"):
            _normalize_fabric_handle_bytes(b"\x00" * 8)

    def test_normalize_fabric_handle_bytes_rejects_unconvertible_objects(self):
        class Unconvertible:
            def __bytes__(self):
                raise TypeError("boom")

        with pytest.raises(CudaFabricError, match="Unable to convert"):
            _normalize_fabric_handle_bytes(Unconvertible())


class TestAmdFabricHelpers:
    def test_round_up(self):
        assert _amd_round_up(1, 64) == 64
        assert _amd_round_up(64, 64) == 64
        assert _amd_round_up(65, 64) == 128

    def test_round_up_rejects_nonpositive_granularity(self):
        with pytest.raises(ValueError, match="granularity must be > 0"):
            _amd_round_up(1, 0)

    def test_normalize_fabric_handle_bytes(self):
        original = bytes(range(AMD_FABRIC_HANDLE_BYTES))
        assert _normalize_amd_fabric_handle_bytes(original) == original
        assert _normalize_amd_fabric_handle_bytes(bytearray(original)) == original
        assert _normalize_amd_fabric_handle_bytes(memoryview(original)) == original
        tensor = torch.tensor(list(original), dtype=torch.uint8)
        assert _normalize_amd_fabric_handle_bytes(tensor) == original

    def test_normalize_fabric_handle_bytes_rejects_wrong_size(self):
        with pytest.raises(AmdFabricError, match="expected 64 bytes"):
            _normalize_amd_fabric_handle_bytes(b"\x00" * 8)

    def test_amdsmi_bdf_from_pci_bus_id(self):
        bdf = _amdsmi_bdf_from_pci_bus_id("GPU 0000:41:00.0")
        assert bdf.as_uint == 0x4100
        assert bdf.bdf.domain_number == 0
        assert bdf.bdf.bus_number == 0x41
        assert bdf.bdf.device_number == 0
        assert bdf.bdf.function_number == 0

    def test_amdsmi_bdf_from_pci_bus_id_rejects_invalid_string(self):
        with pytest.raises(AmdFabricError, match="Unable to parse"):
            _amdsmi_bdf_from_pci_bus_id("not-a-bdf")

    def test_valid_fabric_domain_rejects_sentinels(self):
        assert _valid_fabric_domain(bytes(range(16)), 7)
        assert not _valid_fabric_domain(bytes(16), 7)
        assert not _valid_fabric_domain(bytes([0x99] * 16), 7)
        assert not _valid_fabric_domain(bytes(range(16)), 0xFFFFFFFF)


class TestAmdFabricDriver:
    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("allocate_exportable", (4096,)),
            ("export_handle", (ExportableMemory(va=0, size=0, allocation=LocalAllocation(va=0, size=0, handle=0)),)),
            ("import_and_map", (0, b"\x00" * AMD_FABRIC_HANDLE_BYTES, 4096)),
            (
                "cleanup",
                (
                    PeerMapping(
                        peer_rank=0,
                        transport=InterconnectLevel.INTRA_RACK_FABRIC,
                        remote_va=0,
                        size=0,
                    ),
                ),
            ),
            ("cleanup", (LocalAllocation(va=0, size=0, handle=0),)),
        ],
    )
    def test_public_methods_require_initialize(self, method_name, args):
        driver = AmdFabricDriver()
        method = getattr(driver, method_name)
        with pytest.raises(AmdFabricError, match="not initialized"):
            method(*args)

    def test_initialize_raises_when_no_hip_runtime(self, monkeypatch):
        monkeypatch.setattr(amd_driver_module, "_hip", None)
        driver = AmdFabricDriver()
        with pytest.raises(AmdFabricNotSupported, match="libamdhip64.so.*not found"):
            driver.initialize(0)

    def test_cleanup_import_target_attempts_all_cleanup_steps(self, monkeypatch):
        calls = []

        class FakeHipDriver:
            def hipMemUnmap(self, remote_va, size):
                calls.append(("unmap", remote_va.value, size))
                return 1

            def hipMemRelease(self, handle):
                calls.append(("release", handle.value))
                return 0

            def hipMemAddressFree(self, remote_va, size):
                calls.append(("free", remote_va.value, size))
                return 0

        monkeypatch.setattr(amd_driver_module, "_hip", FakeHipDriver())
        driver = AmdFabricDriver()
        driver._initialized = True
        mapping = PeerMapping(
            peer_rank=2,
            transport=InterconnectLevel.INTRA_RACK_FABRIC,
            remote_va=0x2000,
            size=4096,
            _driver_handle=99,
        )

        with pytest.raises(AmdFabricError, match="hipMemUnmap"):
            driver.cleanup(mapping)

        assert calls == [
            ("unmap", 0x2000, 4096),
            ("release", 99),
            ("free", 0x2000, 4096),
        ]

    def test_cleanup_local_target_attempts_all_cleanup_steps(self, monkeypatch):
        calls = []

        class FakeHipDriver:
            def hipMemUnmap(self, va, size):
                calls.append(("unmap", va.value, size))
                return 1

            def hipMemRelease(self, handle):
                calls.append(("release", handle.value))
                return 0

            def hipMemAddressFree(self, va, size):
                calls.append(("free", va.value, size))
                return 0

        monkeypatch.setattr(amd_driver_module, "_hip", FakeHipDriver())
        driver = AmdFabricDriver()
        driver._initialized = True
        allocation = LocalAllocation(va=0x1000, size=8192, handle=77)

        with pytest.raises(AmdFabricError, match="hipMemUnmap"):
            driver.cleanup(allocation)

        assert calls == [
            ("unmap", 0x1000, 8192),
            ("release", 77),
            ("free", 0x1000, 8192),
        ]


class TestNvidiaFabricDriver:
    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("allocate_exportable", (4096,)),
            ("export_handle", (ExportableMemory(va=0, size=0, allocation=LocalAllocation(va=0, size=0, handle=0)),)),
            ("import_and_map", (0, b"\x00" * FABRIC_HANDLE_BYTES, 4096)),
            (
                "cleanup",
                (
                    PeerMapping(
                        peer_rank=0,
                        transport=InterconnectLevel.INTRA_RACK_FABRIC,
                        remote_va=0,
                        size=0,
                    ),
                ),
            ),
            ("cleanup", (LocalAllocation(va=0, size=0, handle=0),)),
        ],
    )
    def test_public_methods_require_initialize(self, method_name, args):
        driver = NvidiaFabricDriver()
        method = getattr(driver, method_name)
        with pytest.raises(CudaFabricError, match="not initialized"):
            method(*args)

    def test_initialize_raises_when_no_cuda_driver(self, monkeypatch):
        monkeypatch.setattr(nvidia_driver_module, "_cuda_driver", None)
        driver = NvidiaFabricDriver()
        with pytest.raises(CudaFabricNotSupported, match="libcuda.so.*not found"):
            driver.initialize(0)

    def test_initialize_raises_not_supported_for_missing_required_symbol(self, monkeypatch):
        class IncompleteCudaDriver:
            pass

        monkeypatch.setattr(nvidia_driver_module, "_cuda_driver", IncompleteCudaDriver())
        driver = NvidiaFabricDriver()
        with pytest.raises(CudaFabricNotSupported, match="missing required VMM symbol: cuInit"):
            driver.initialize(0)

    def test_cleanup_import_target_attempts_all_cleanup_steps(self, monkeypatch):
        calls = []

        class FakeCudaDriver:
            def cuMemUnmap(self, remote_va, size):
                calls.append(("unmap", remote_va, size))
                return 1

            def cuMemRelease(self, handle):
                calls.append(("release", handle))
                return 0

            def cuMemAddressFree(self, remote_va, size):
                calls.append(("free", remote_va, size))
                return 0

        monkeypatch.setattr(nvidia_driver_module, "_cuda_driver", FakeCudaDriver())
        driver = NvidiaFabricDriver()
        driver._initialized = True
        mapping = PeerMapping(
            peer_rank=2,
            transport=InterconnectLevel.INTRA_RACK_FABRIC,
            remote_va=0x2000,
            size=4096,
            _driver_handle=99,
        )

        with pytest.raises(CudaFabricError, match="cuMemUnmap"):
            driver.cleanup(mapping)

        assert calls == [
            ("unmap", 0x2000, 4096),
            ("release", 99),
            ("free", 0x2000, 4096),
        ]

    def test_cleanup_local_target_attempts_all_cleanup_steps(self, monkeypatch):
        calls = []

        class FakeCudaDriver:
            def cuMemUnmap(self, va, size):
                calls.append(("unmap", va, size))
                return 1

            def cuMemRelease(self, handle):
                calls.append(("release", handle))
                return 0

            def cuMemAddressFree(self, va, size):
                calls.append(("free", va, size))
                return 0

        monkeypatch.setattr(nvidia_driver_module, "_cuda_driver", FakeCudaDriver())
        driver = NvidiaFabricDriver()
        driver._initialized = True
        allocation = LocalAllocation(va=0x1000, size=8192, handle=77)

        with pytest.raises(CudaFabricError, match="cuMemUnmap"):
            driver.cleanup(allocation)

        assert calls == [
            ("unmap", 0x1000, 8192),
            ("release", 77),
            ("free", 0x1000, 8192),
        ]
