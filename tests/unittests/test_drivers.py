# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the driver layer.

These are pure unit tests and should run without GPUs or a distributed setup.
"""

from __future__ import annotations

import importlib.util

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
from iris.drivers.local import amd as local_amd_driver_module
from iris.drivers.local.amd import LocalHipDriver
from iris.drivers.fabric.nvidia import (
    CudaFabricError,
    CudaFabricNotSupported,
    FABRIC_HANDLE_BYTES,
    NvidiaFabricDriver,
    _normalize_fabric_handle_bytes,
    _round_up,
)
from iris.host.distributed.topology import InterconnectLevel
from iris.host.platform import libs as platform_libs


@pytest.mark.parametrize("unversioned_available", [True, False])
def test_amd_fabric_library_selection_with_stale_system_soname(monkeypatch, unversioned_available):
    """An SDK library takes precedence over a different major version in ldconfig."""
    loaded = {}

    def load(name):
        if not unversioned_available and name.endswith(".so"):
            raise OSError("No development symlink")
        return loaded.setdefault(name, object())

    monkeypatch.setattr(amd_driver_module.ctypes, "CDLL", load)
    monkeypatch.setattr(amd_driver_module.ctypes.util, "find_library", lambda name: f"lib{name}.so.5")
    monkeypatch.setattr(platform_libs, "_mapped_library_paths", lambda stem: [])
    spec = importlib.util.spec_from_file_location("_iris_amd_library_selection_test", amd_driver_module.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suffix = ".so" if unversioned_available else ".so.5"
    assert module._hip is loaded["libamdhip64" + suffix]
    assert module._amdsmi is loaded["libamd_smi" + suffix]
    if unversioned_available:
        assert "libamdhip64.so.5" not in loaded


def test_amd_fabric_library_selection_prefers_loaded_copy(monkeypatch):
    """The HIP runtime and amd_smi torch already loaded win over any name lookup."""
    loaded = {}
    mapped = {
        "libamdhip64": ["/sdk/core/lib/libamdhip64.so.7"],
        "libamd_smi": ["/sdk/core/lib/libamd_smi.so.27"],
    }

    monkeypatch.setattr(amd_driver_module.ctypes, "CDLL", lambda name: loaded.setdefault(name, object()))
    monkeypatch.setattr(amd_driver_module.ctypes.util, "find_library", lambda name: f"lib{name}.so.5")
    monkeypatch.setattr(platform_libs, "_mapped_library_paths", lambda stem: mapped[stem])
    spec = importlib.util.spec_from_file_location("_iris_amd_loaded_library_test", amd_driver_module.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._hip is loaded["/sdk/core/lib/libamdhip64.so.7"]
    assert module._amdsmi is loaded["/sdk/core/lib/libamd_smi.so.27"]
    assert set(loaded) == {"/sdk/core/lib/libamdhip64.so.7", "/sdk/core/lib/libamd_smi.so.27"}


class _FakeVaHip:
    """Stand-in for the HIP VA ctypes binding used by reserve_va/free_va tests.

    hipMemAddressReserve writes the chosen raw base into the caller's c_void_p and
    records the (size, alignment) it was asked for; hipMemAddressFree records the
    (ptr, size) it was asked to release. This lets the tests assert the over-reserve
    size, the trimmed base, and that the ORIGINAL reservation is what gets freed,
    without a GPU.
    """

    def __init__(self, raw_base, hip_success):
        self._raw_base = raw_base
        self._hip_success = hip_success
        self.reserve_calls = []
        self.free_calls = []

    def hipMemAddressReserve(self, pptr, size, alignment, addr, flags):
        self.reserve_calls.append((size, alignment))
        pptr._obj.value = self._raw_base
        return self._hip_success

    def hipMemAddressFree(self, ptr, size):
        self.free_calls.append((ptr.value, size))
        return self._hip_success


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

    def test_granularity_methods_route_distinct_flags(self, monkeypatch):
        """get_minimum_granularity/get_recommended_granularity must query with the matching flag.

        A flag mix-up (easy to introduce while refactoring the shared _query_granularity
        helper) would silently make the allocator back memory with the wrong granule, or
        report the wrong minimum in get_stats(). Returning distinct values per flag here
        means this test fails if the two public methods ever route to the same flag or swap
        them, which a test that only checks "some positive int came back" would not catch.
        """
        driver = AmdFabricDriver()
        driver._initialized = True
        seen_flags = []

        def fake_query_granularity(granularity_flag):
            seen_flags.append(granularity_flag)
            return {
                amd_driver_module.hipMemAllocationGranularityMinimum: 4096,
                amd_driver_module.hipMemAllocationGranularityRecommended: 2 * 1024 * 1024,
            }[granularity_flag]

        monkeypatch.setattr(driver, "_query_granularity", fake_query_granularity)

        assert driver.get_minimum_granularity() == 4096
        assert driver.get_recommended_granularity() == 2 * 1024 * 1024
        assert seen_flags == [
            amd_driver_module.hipMemAllocationGranularityMinimum,
            amd_driver_module.hipMemAllocationGranularityRecommended,
        ]


    def test_reserve_va_over_aligns_beyond_granule_and_frees_raw(self, monkeypatch):
        """AmdFabricDriver shares the over-align/trim contract with LocalHipDriver.

        The fabric driver keeps its own copy of reserve_va/free_va, so verify the trim
        and raw-release behavior here too to catch drift between the two copies.
        """
        granularity = 2 * 1024 * 1024
        alignment = 8 * 1024 * 1024
        size = 4 * 1024 * 1024
        raw_base = granularity

        driver = AmdFabricDriver()
        driver._initialized = True
        driver._granularity = granularity
        fake = _FakeVaHip(raw_base, amd_driver_module.HIP_SUCCESS)
        monkeypatch.setattr(amd_driver_module, "_hip", fake)

        va = driver.reserve_va(size, alignment)
        assert va % alignment == 0
        assert fake.reserve_calls == [(size + alignment, granularity)]
        assert driver._aligned_reservations == {va: (raw_base, size + alignment)}

        driver.free_va(va, size)
        assert fake.free_calls == [(raw_base, size + alignment)]
        assert va not in driver._aligned_reservations


class TestLocalHipDriver:
    def test_granularity_methods_route_distinct_flags(self, monkeypatch):
        """Same contract as AmdFabricDriver: minimum and recommended must use distinct flags."""
        driver = LocalHipDriver()
        driver._initialized = True
        seen_flags = []

        def fake_query_granularity(granularity_flag):
            seen_flags.append(granularity_flag)
            return {
                local_amd_driver_module.hipMemAllocationGranularityMinimum: 4096,
                local_amd_driver_module.hipMemAllocationGranularityRecommended: 2 * 1024 * 1024,
            }[granularity_flag]

        monkeypatch.setattr(driver, "_query_granularity", fake_query_granularity)

        assert driver.get_minimum_granularity() == 4096
        assert driver.get_recommended_granularity() == 2 * 1024 * 1024
        assert seen_flags == [
            local_amd_driver_module.hipMemAllocationGranularityMinimum,
            local_amd_driver_module.hipMemAllocationGranularityRecommended,
        ]

    def test_initialize_resets_cached_granularity(self, monkeypatch):
        """Re-initialize must drop cached granularity so a device switch re-queries the driver.

        get_minimum_granularity/get_recommended_granularity memoize their first query. If
        initialize() did not clear those caches, binding the driver to a different device would
        keep reporting the old device's granularity.
        """
        driver = LocalHipDriver()
        driver._initialized = True
        driver._min_granularity = 4096
        driver._granularity = 2 * 1024 * 1024

        class FakeHip:
            def hipSetDevice(self, ordinal):
                return local_amd_driver_module.HIP_SUCCESS

        monkeypatch.setattr(local_amd_driver_module, "_hip", FakeHip())
        monkeypatch.setattr(local_amd_driver_module, "_configure_signatures", lambda: None)

        driver.initialize(1)

        assert driver._min_granularity is None
        assert driver._granularity is None

    def test_reserve_va_over_aligns_beyond_granule_and_frees_raw(self, monkeypatch):
        """reserve_va must honor an alignment larger than the granule.

        hipMemAddressReserve only guarantees granularity alignment on this stack, so
        reserve_va over-reserves by ``alignment`` bytes and trims to the first aligned
        address. The symmetric heap relies on this to give every rank a base aligned to
        MAX_ALIGNMENT; without it an allocation's aligned heap offset would not translate
        to an aligned pointer. free_va must then release the ORIGINAL (raw) reservation,
        not the trimmed base, or the driver leaks VA and HIP rejects the free.
        """
        granularity = 2 * 1024 * 1024  # 2 MiB
        alignment = 8 * 1024 * 1024  # 8 MiB
        size = 4 * 1024 * 1024  # 4 MiB
        raw_base = granularity  # 2 MiB-aligned but NOT 8 MiB-aligned -> must be trimmed

        driver = LocalHipDriver()
        driver._initialized = True
        driver._granularity = granularity
        fake = _FakeVaHip(raw_base, local_amd_driver_module.HIP_SUCCESS)
        monkeypatch.setattr(local_amd_driver_module, "_hip", fake)

        va = driver.reserve_va(size, alignment)

        # Trimmed to the first alignment-aligned address inside the over-reservation.
        assert va % alignment == 0
        assert raw_base <= va < raw_base + alignment
        # Over-reserved by `alignment` extra bytes, reserved at granule alignment.
        assert fake.reserve_calls == [(size + alignment, granularity)]
        assert driver._aligned_reservations == {va: (raw_base, size + alignment)}

        driver.free_va(va, size)

        # The ORIGINAL reservation (raw base, raw size) is released, and the bookkeeping
        # entry is dropped so a stale lookup cannot mis-free later.
        assert fake.free_calls == [(raw_base, size + alignment)]
        assert va not in driver._aligned_reservations

    def test_reserve_va_within_granule_is_not_trimmed(self, monkeypatch):
        """When alignment <= granularity the driver already satisfies it directly.

        reserve_va must then reserve exactly ``size`` (no over-reservation) and record no
        trimming bookkeeping, so free_va releases exactly what was reserved.
        """
        granularity = 2 * 1024 * 1024
        size = 4 * 1024 * 1024
        raw_base = granularity

        driver = LocalHipDriver()
        driver._initialized = True
        driver._granularity = granularity
        fake = _FakeVaHip(raw_base, local_amd_driver_module.HIP_SUCCESS)
        monkeypatch.setattr(local_amd_driver_module, "_hip", fake)

        va = driver.reserve_va(size, granularity)

        assert va == raw_base
        assert fake.reserve_calls == [(size, granularity)]
        assert driver._aligned_reservations == {}

        driver.free_va(va, size)
        assert fake.free_calls == [(raw_base, size)]
        assert driver._aligned_reservations == {}


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
