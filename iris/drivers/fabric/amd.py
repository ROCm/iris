# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
AMD HIP VMM fabric driver.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import re
from collections.abc import Callable
from typing import Any, Optional

import torch

from iris.drivers.base import (
    BaseDriver,
    DriverError,
    DriverNotSupported,
    ExportableMemory,
    LocalAllocation,
    MappingPlacement,
    PeerMapping,
)
from iris.host.distributed.topology import InterconnectLevel

logger = logging.getLogger("iris.drivers.fabric.amd")

__all__ = [
    "AmdFabricError",
    "AmdFabricNotSupported",
    "AMD_FABRIC_HANDLE_BYTES",
    "FABRIC_HANDLE_BYTES",
    "AmdFabricDriver",
]


def _load_cdll(*names: Optional[str]) -> Any:
    for name in names:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_hip = _load_cdll(
    ctypes.util.find_library("amdhip64"),
    "libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so",
)
_amdsmi = _load_cdll(
    ctypes.util.find_library("amd_smi"),
    "libamd_smi.so",
    "/opt/rocm/lib/libamd_smi.so",
)

HIP_SUCCESS = 0
HIP_ERROR_NOT_SUPPORTED = 801

AMDSMI_STATUS_SUCCESS = 0
AMDSMI_STATUS_NOT_SUPPORTED = 2
AMDSMI_STATUS_NOT_YET_IMPLEMENTED = 3
AMDSMI_STATUS_NO_DATA = 40
AMDSMI_INIT_AMD_GPUS = 1 << 1

FABRIC_HANDLE_BYTES = 64
AMD_FABRIC_HANDLE_BYTES = FABRIC_HANDLE_BYTES

_UINT32_MAX = 0xFFFFFFFF
_AMDSMI_FABRIC_PPOD_SENTINEL = bytes([0x99] * 16)
_PCI_BUS_ID_BYTES = 256

hipMemAllocationTypePinned = 0x1
hipMemHandleTypeFabric = 0x8
hipMemLocationTypeDevice = 0x1
hipMemAllocationGranularityRecommended = 0x1
hipMemAccessFlagsProtReadWrite = 0x3

hipMemGenericAllocationHandle_t = ctypes.c_void_p
amdsmi_processor_handle = ctypes.c_void_p
amdsmi_status_t = ctypes.c_uint32


class AmdFabricError(DriverError):
    """AMD HIP/AMDSMI fabric operation failed."""


class AmdFabricNotSupported(DriverNotSupported):
    """The local AMD stack does not support fabric handles."""


class hipMemLocation(ctypes.Structure):
    """Structure describing a HIP memory location."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _hipMemAllocationHandleTypes(ctypes.Union):
    _fields_ = [
        ("requestedHandleType", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
    ]


class _hipMemAllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class hipMemAllocationProp(ctypes.Structure):
    """Properties for a HIP VMem allocation."""

    _anonymous_ = ("handleTypes",)
    _fields_ = [
        ("type", ctypes.c_int),
        ("handleTypes", _hipMemAllocationHandleTypes),
        ("location", hipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _hipMemAllocationFlags),
    ]


class hipMemAccessDesc(ctypes.Structure):
    """Access descriptor for a HIP VMem mapping."""

    _fields_ = [
        ("location", hipMemLocation),
        ("flags", ctypes.c_int),
    ]


class hipMemFabricHandle(ctypes.Structure):
    """HIP fabric handle. ROCm defines this with HIP_IPC_HANDLE_SIZE bytes."""

    _fields_ = [("data", ctypes.c_ubyte * FABRIC_HANDLE_BYTES)]


class _amdsmi_bdf_bitfields(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("function_number", ctypes.c_uint64, 3),
        ("device_number", ctypes.c_uint64, 5),
        ("bus_number", ctypes.c_uint64, 8),
        ("domain_number", ctypes.c_uint64, 48),
    ]


class amdsmi_bdf_t(ctypes.Union):
    _pack_ = 1
    _fields_ = [
        ("bdf", _amdsmi_bdf_bitfields),
        ("as_uint", ctypes.c_uint64),
    ]


class amdsmi_fabric_info_v1_t(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("accelerator_id", ctypes.c_uint32),
        ("fabric_type", ctypes.c_uint32),
        ("bandwidth", ctypes.c_uint32),
        ("latency", ctypes.c_uint32),
        ("ppod_id", ctypes.c_ubyte * 16),
        ("ppod_size", ctypes.c_uint32),
        ("vpod_id", ctypes.c_uint32),
        ("vpod_size", ctypes.c_uint32),
        ("vpod_active_accelerators", ctypes.c_uint32 * 32),
        ("local_accelerators", ctypes.c_uint32 * 16),
        ("addr_mode", ctypes.c_uint32),
        ("accel_state", ctypes.c_uint32),
    ]


class _amdsmi_fabric_info_union(ctypes.Union):
    _pack_ = 1
    _fields_ = [("v1", amdsmi_fabric_info_v1_t)]


class amdsmi_fabric_info_ver_t(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("fabric_version", _amdsmi_fabric_info_union),
    ]


class amdsmi_fabric_info_t(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("bdf", amdsmi_bdf_t),
        ("fabric_info", amdsmi_fabric_info_ver_t),
        ("reserved", ctypes.c_uint32 * 15),
        ("_padding", ctypes.c_ubyte * 4),
    ]


def _get_required_hip_symbol(name: str) -> Any:
    if _hip is None:
        raise AmdFabricNotSupported("libamdhip64.so not found")

    symbol = getattr(_hip, name, None)
    if symbol is None:
        raise AmdFabricNotSupported(f"HIP runtime missing required symbol: {name}")
    return symbol


def _get_required_amdsmi_symbol(name: str) -> Any:
    if _amdsmi is None:
        raise AmdFabricNotSupported("libamd_smi.so not found")

    symbol = getattr(_amdsmi, name, None)
    if symbol is None:
        raise AmdFabricNotSupported(f"AMDSMI library missing required symbol: {name}")
    return symbol


def _configure_signatures() -> None:
    """Configure ctypes signatures for all HIP and AMDSMI functions used here."""
    if _hip is None:
        return

    hip_set_device = _get_required_hip_symbol("hipSetDevice")
    hip_device_get_pci_bus_id = _get_required_hip_symbol("hipDeviceGetPCIBusId")
    hip_mem_get_allocation_granularity = _get_required_hip_symbol("hipMemGetAllocationGranularity")
    hip_mem_create = _get_required_hip_symbol("hipMemCreate")
    hip_mem_address_reserve = _get_required_hip_symbol("hipMemAddressReserve")
    hip_mem_map = _get_required_hip_symbol("hipMemMap")
    hip_mem_set_access = _get_required_hip_symbol("hipMemSetAccess")
    hip_mem_unmap = _get_required_hip_symbol("hipMemUnmap")
    hip_mem_release = _get_required_hip_symbol("hipMemRelease")
    hip_mem_address_free = _get_required_hip_symbol("hipMemAddressFree")
    hip_mem_export_to_shareable_handle = _get_required_hip_symbol("hipMemExportToShareableHandle")
    hip_mem_import_from_shareable_handle = _get_required_hip_symbol("hipMemImportFromShareableHandle")
    hip_get_error_string = _get_required_hip_symbol("hipGetErrorString")

    hip_set_device.argtypes = [ctypes.c_int]
    hip_set_device.restype = ctypes.c_int

    hip_device_get_pci_bus_id.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    hip_device_get_pci_bus_id.restype = ctypes.c_int

    hip_mem_get_allocation_granularity.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(hipMemAllocationProp),
        ctypes.c_int,
    ]
    hip_mem_get_allocation_granularity.restype = ctypes.c_int

    hip_mem_create.argtypes = [
        ctypes.POINTER(hipMemGenericAllocationHandle_t),
        ctypes.c_size_t,
        ctypes.POINTER(hipMemAllocationProp),
        ctypes.c_ulonglong,
    ]
    hip_mem_create.restype = ctypes.c_int

    hip_mem_address_reserve.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
    ]
    hip_mem_address_reserve.restype = ctypes.c_int

    hip_mem_map.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        hipMemGenericAllocationHandle_t,
        ctypes.c_ulonglong,
    ]
    hip_mem_map.restype = ctypes.c_int

    hip_mem_set_access.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(hipMemAccessDesc),
        ctypes.c_size_t,
    ]
    hip_mem_set_access.restype = ctypes.c_int

    hip_mem_unmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    hip_mem_unmap.restype = ctypes.c_int

    hip_mem_release.argtypes = [hipMemGenericAllocationHandle_t]
    hip_mem_release.restype = ctypes.c_int

    hip_mem_address_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    hip_mem_address_free.restype = ctypes.c_int

    hip_mem_export_to_shareable_handle.argtypes = [
        ctypes.c_void_p,
        hipMemGenericAllocationHandle_t,
        ctypes.c_int,
        ctypes.c_ulonglong,
    ]
    hip_mem_export_to_shareable_handle.restype = ctypes.c_int

    hip_mem_import_from_shareable_handle.argtypes = [
        ctypes.POINTER(hipMemGenericAllocationHandle_t),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    hip_mem_import_from_shareable_handle.restype = ctypes.c_int

    hip_get_error_string.argtypes = [ctypes.c_int]
    hip_get_error_string.restype = ctypes.c_char_p

    if _amdsmi is None:
        return

    amdsmi_init = _get_required_amdsmi_symbol("amdsmi_init")
    amdsmi_get_processor_handle_from_bdf = _get_required_amdsmi_symbol("amdsmi_get_processor_handle_from_bdf")
    amdsmi_get_gpu_fabric_info = _get_required_amdsmi_symbol("amdsmi_get_gpu_fabric_info")

    amdsmi_init.argtypes = [ctypes.c_uint64]
    amdsmi_init.restype = amdsmi_status_t

    amdsmi_get_processor_handle_from_bdf.argtypes = [
        amdsmi_bdf_t,
        ctypes.POINTER(amdsmi_processor_handle),
    ]
    amdsmi_get_processor_handle_from_bdf.restype = amdsmi_status_t

    amdsmi_get_gpu_fabric_info.argtypes = [
        amdsmi_processor_handle,
        ctypes.POINTER(amdsmi_fabric_info_t),
    ]
    amdsmi_get_gpu_fabric_info.restype = amdsmi_status_t

    amdsmi_status_code_to_string = getattr(_amdsmi, "amdsmi_status_code_to_string", None)
    if amdsmi_status_code_to_string is not None:
        amdsmi_status_code_to_string.argtypes = [
            amdsmi_status_t,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        amdsmi_status_code_to_string.restype = amdsmi_status_t


def _hip_try(err: int, op_name: str = "HIP operation") -> None:
    """Check a HIP runtime return code and raise a driver exception on error."""
    if err == HIP_SUCCESS:
        return

    error_string = str(err)
    if _hip is not None and hasattr(_hip, "hipGetErrorString"):
        decoded = _hip.hipGetErrorString(ctypes.c_int(err))
        if decoded:
            error_string = decoded.decode("utf-8", errors="replace")

    message = f"{op_name} failed with HIP error code {err}: {error_string}"
    if err == HIP_ERROR_NOT_SUPPORTED:
        raise AmdFabricNotSupported(message)
    raise AmdFabricError(message)


def _amdsmi_status_string(err: int) -> str:
    if _amdsmi is not None and hasattr(_amdsmi, "amdsmi_status_code_to_string"):
        ptr = ctypes.c_char_p()
        ret = _amdsmi.amdsmi_status_code_to_string(amdsmi_status_t(err), ctypes.byref(ptr))
        if ret == AMDSMI_STATUS_SUCCESS and ptr.value:
            return ptr.value.decode("utf-8", errors="replace")
    return str(err)


def _amdsmi_try(err: int, op_name: str = "AMDSMI operation") -> None:
    """Check an AMDSMI return code and raise a driver exception on error."""
    if err == AMDSMI_STATUS_SUCCESS:
        return

    message = f"{op_name} failed with AMDSMI status {err}: {_amdsmi_status_string(err)}"
    if err in (AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_DATA):
        raise AmdFabricNotSupported(message)
    raise AmdFabricError(message)


def _round_up(value: int, granularity: int) -> int:
    if granularity <= 0:
        raise ValueError(f"granularity must be > 0, got {granularity}")
    return ((value + granularity - 1) // granularity) * granularity


def _normalize_fabric_handle_bytes(raw_handle: Any) -> bytes:
    if isinstance(raw_handle, memoryview):
        data = raw_handle.tobytes()
    elif isinstance(raw_handle, (bytes, bytearray)):
        data = bytes(raw_handle)
    elif isinstance(raw_handle, torch.Tensor):
        data = bytes(raw_handle.detach().to("cpu", copy=True).flatten().tolist())
    else:
        try:
            data = bytes(raw_handle)
        except Exception:
            try:
                data = ctypes.string_at(ctypes.addressof(raw_handle), FABRIC_HANDLE_BYTES)
            except Exception as exc:
                raise AmdFabricError("Unable to convert fabric handle object to bytes") from exc

    if len(data) != FABRIC_HANDLE_BYTES:
        raise AmdFabricError(f"Fabric handle serialization expected {FABRIC_HANDLE_BYTES} bytes, got {len(data)}")
    return data


def _run_cleanup_steps(*steps: tuple[str, Callable[[], None]]) -> None:
    first_error = None
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            else:
                logger.warning("Secondary cleanup step %s failed: %s", name, exc)
    if first_error is not None:
        raise first_error


def _cleanup_after_failure(*steps: tuple[str, Callable[[], None]]) -> None:
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            logger.warning("Cleanup step %s failed after earlier failure: %s", name, exc)


_PCI_BDF_RE = re.compile(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])")


def _amdsmi_bdf_from_pci_bus_id(pci_bus_id: str) -> amdsmi_bdf_t:
    match = _PCI_BDF_RE.search(pci_bus_id.strip())
    if match is None:
        raise AmdFabricError(f"Unable to parse HIP PCI bus ID: {pci_bus_id!r}")

    domain = int(match.group(1), 16)
    bus = int(match.group(2), 16)
    device = int(match.group(3), 16)
    function = int(match.group(4), 16)

    bdf = amdsmi_bdf_t()
    bdf.as_uint = (domain << 16) | (bus << 8) | (device << 3) | function
    return bdf


def _valid_fabric_domain(ppod_id: bytes, vpod_id: int) -> bool:
    if not ppod_id or ppod_id == bytes(len(ppod_id)) or ppod_id == _AMDSMI_FABRIC_PPOD_SENTINEL:
        return False
    return vpod_id != _UINT32_MAX


class AmdFabricDriver(BaseDriver):
    """
    AMD HIP VMM fabric driver.

    Uses HIP fabric handles for cross-host VMM export/import. Handle exchange is
    performed by higher layers using torch.distributed collectives; this driver
    only creates, exports, imports, maps, and releases VMM allocations.
    """

    def __init__(self) -> None:
        self._device_ordinal: int = 0
        self._granularity: Optional[int] = None
        self._initialized: bool = False
        self._fabric_domain: Optional[tuple[str, int]] = None

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise AmdFabricError("AmdFabricDriver not initialized - call initialize() first")

    def _make_alloc_props(self) -> hipMemAllocationProp:
        props = hipMemAllocationProp()
        props.type = hipMemAllocationTypePinned
        props.requestedHandleTypes = hipMemHandleTypeFabric
        props.location.type = hipMemLocationTypeDevice
        props.location.id = self._device_ordinal
        props.win32HandleMetaData = None
        return props

    def _get_granularity(self) -> int:
        if self._granularity is not None:
            return self._granularity

        props = self._make_alloc_props()
        granularity = ctypes.c_size_t()
        _hip_try(
            _hip.hipMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(props),
                hipMemAllocationGranularityRecommended,
            ),
            "hipMemGetAllocationGranularity",
        )
        self._granularity = int(granularity.value)
        return self._granularity

    def _mem_set_access(self, va: int, size: int) -> None:
        desc = hipMemAccessDesc()
        desc.location.type = hipMemLocationTypeDevice
        desc.location.id = self._device_ordinal
        desc.flags = hipMemAccessFlagsProtReadWrite
        _hip_try(
            _hip.hipMemSetAccess(ctypes.c_void_p(va), size, ctypes.byref(desc), 1),
            "hipMemSetAccess",
        )

    def _get_device_pci_bus_id(self) -> str:
        pci_bus_id = ctypes.create_string_buffer(_PCI_BUS_ID_BYTES)
        _hip_try(
            _hip.hipDeviceGetPCIBusId(pci_bus_id, _PCI_BUS_ID_BYTES, self._device_ordinal),
            "hipDeviceGetPCIBusId",
        )
        return pci_bus_id.value.decode("utf-8", errors="replace")

    def _query_fabric_domain(self) -> tuple[str, int]:
        if _amdsmi is None:
            raise AmdFabricNotSupported("libamd_smi.so not found")

        _amdsmi_try(_amdsmi.amdsmi_init(AMDSMI_INIT_AMD_GPUS), "amdsmi_init")

        pci_bus_id = self._get_device_pci_bus_id()
        bdf = _amdsmi_bdf_from_pci_bus_id(pci_bus_id)
        processor = amdsmi_processor_handle()
        _amdsmi_try(
            _amdsmi.amdsmi_get_processor_handle_from_bdf(bdf, ctypes.byref(processor)),
            "amdsmi_get_processor_handle_from_bdf",
        )
        if processor.value is None:
            raise AmdFabricError(f"AMDSMI returned a null processor handle for PCI bus ID {pci_bus_id}")

        fabric_info = amdsmi_fabric_info_t()
        err = int(_amdsmi.amdsmi_get_gpu_fabric_info(processor, ctypes.byref(fabric_info)))
        if err == AMDSMI_STATUS_NO_DATA:
            raise AmdFabricNotSupported(f"AMDSMI reported no GPU fabric data for PCI bus ID {pci_bus_id}")
        _amdsmi_try(err, "amdsmi_get_gpu_fabric_info")

        v1 = fabric_info.fabric_info.fabric_version.v1
        ppod_id = bytes(v1.ppod_id)
        vpod_id = int(v1.vpod_id)
        if not _valid_fabric_domain(ppod_id, vpod_id):
            raise AmdFabricNotSupported(
                f"AMDSMI fabric domain is unavailable for PCI bus ID {pci_bus_id}: "
                f"ppod_id={ppod_id.hex()} vpod_id={vpod_id}"
            )

        return ppod_id.hex(), vpod_id

    def initialize(self, device_ordinal: int) -> None:
        """Prepare HIP VMM and verify that the selected GPU has fabric data."""
        if _hip is None:
            raise AmdFabricNotSupported("libamdhip64.so not found")
        if _amdsmi is None:
            raise AmdFabricNotSupported("libamd_smi.so not found")

        _configure_signatures()
        _hip_try(_hip.hipSetDevice(device_ordinal), "hipSetDevice")
        self._device_ordinal = device_ordinal
        self._granularity = None
        self._fabric_domain = self._query_fabric_domain()
        self._initialized = True
        logger.info(
            "AmdFabricDriver initialized (device %d, fabric_domain=%s:%d)",
            device_ordinal,
            self._fabric_domain[0],
            self._fabric_domain[1],
        )

    def allocate_exportable(
        self,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> LocalAllocation:
        """
        Allocate HIP VMem memory exportable as a fabric handle.

        If placement is supplied, the caller must already own a sufficiently
        large, granularity-aligned VA range containing [placement.va,
        placement.va + size).
        """
        self._check_initialized()
        props = self._make_alloc_props()
        granularity = self._get_granularity()
        alloc_size = _round_up(size, granularity)

        reserved_va = placement is None
        mapped_va = int(placement.va) if placement is not None else 0
        handle = hipMemGenericAllocationHandle_t()
        mapped = False

        try:
            if reserved_va:
                reserved = ctypes.c_void_p()
                _hip_try(
                    _hip.hipMemAddressReserve(ctypes.byref(reserved), alloc_size, granularity, None, 0),
                    "hipMemAddressReserve",
                )
                if reserved.value is None:
                    raise AmdFabricError("hipMemAddressReserve returned a null VA")
                mapped_va = int(reserved.value)

            _hip_try(
                _hip.hipMemCreate(ctypes.byref(handle), alloc_size, ctypes.byref(props), 0),
                "hipMemCreate",
            )
            _hip_try(
                _hip.hipMemMap(ctypes.c_void_p(mapped_va), alloc_size, 0, handle, 0),
                "hipMemMap",
            )
            mapped = True
            access_base, access_bytes = (
                placement.access_range(alloc_size, cumulative=True)
                if placement is not None
                else (mapped_va, alloc_size)
            )
            self._mem_set_access(access_base, access_bytes)
            return LocalAllocation(
                va=mapped_va,
                size=alloc_size,
                handle=int(handle.value),
                _va_owned=reserved_va,
            )
        except Exception:
            steps: list[tuple[str, Callable[[], None]]] = []
            if mapped:
                steps.append(
                    (
                        "hipMemUnmap",
                        lambda: _hip_try(
                            _hip.hipMemUnmap(ctypes.c_void_p(mapped_va), alloc_size),
                            "hipMemUnmap",
                        ),
                    )
                )
            if handle.value:
                steps.append(
                    (
                        "hipMemRelease",
                        lambda: _hip_try(
                            _hip.hipMemRelease(hipMemGenericAllocationHandle_t(int(handle.value))),
                            "hipMemRelease",
                        ),
                    )
                )
            if reserved_va and mapped_va:
                steps.append(
                    (
                        "hipMemAddressFree",
                        lambda: _hip_try(
                            _hip.hipMemAddressFree(ctypes.c_void_p(mapped_va), alloc_size),
                            "hipMemAddressFree",
                        ),
                    )
                )
            _cleanup_after_failure(*steps)
            raise

    def export_handle(self, memory: ExportableMemory) -> bytes:
        """Export a 64-byte HIP fabric handle for a driver-created allocation."""
        self._check_initialized()
        if memory.allocation is None:
            raise AmdFabricNotSupported("AMD fabric driver can only export driver-created VMM allocations")

        raw = hipMemFabricHandle()
        _hip_try(
            _hip.hipMemExportToShareableHandle(
                ctypes.byref(raw),
                hipMemGenericAllocationHandle_t(int(memory.allocation.handle)),
                hipMemHandleTypeFabric,
                0,
            ),
            "hipMemExportToShareableHandle",
        )
        return bytes(raw.data)

    def _import_handle(self, handle_bytes: bytes) -> int:
        handle_bytes = _normalize_fabric_handle_bytes(handle_bytes)
        raw = hipMemFabricHandle.from_buffer_copy(handle_bytes)
        imported = hipMemGenericAllocationHandle_t()
        _hip_try(
            _hip.hipMemImportFromShareableHandle(
                ctypes.byref(imported),
                ctypes.byref(raw),
                hipMemHandleTypeFabric,
            ),
            "hipMemImportFromShareableHandle",
        )
        if imported.value is None:
            raise AmdFabricError("hipMemImportFromShareableHandle returned a null allocation handle")
        return int(imported.value)

    def import_and_map(
        self,
        peer_rank: int,
        handle_bytes: bytes,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> PeerMapping:
        """Import a HIP fabric handle and map it into local VMM VA space."""
        self._check_initialized()
        imported_handle = self._import_handle(handle_bytes)

        granularity = self._get_granularity()
        va_owned = placement is None
        mapped_va = int(placement.va) if placement is not None else 0
        mapped = False
        try:
            if va_owned:
                reserved = ctypes.c_void_p()
                _hip_try(
                    _hip.hipMemAddressReserve(ctypes.byref(reserved), size, granularity, None, 0),
                    "hipMemAddressReserve",
                )
                if reserved.value is None:
                    raise AmdFabricError("hipMemAddressReserve returned a null VA")
                mapped_va = int(reserved.value)
            _hip_try(
                _hip.hipMemMap(
                    ctypes.c_void_p(mapped_va),
                    size,
                    0,
                    hipMemGenericAllocationHandle_t(imported_handle),
                    0,
                ),
                "hipMemMap",
            )
            mapped = True
            access_base, access_bytes = (
                placement.access_range(size, cumulative=True) if placement is not None else (mapped_va, size)
            )
            self._mem_set_access(access_base, access_bytes)
        except Exception:
            steps: list[tuple[str, Callable[[], None]]] = []
            if mapped:
                steps.append(
                    (
                        "hipMemUnmap",
                        lambda: _hip_try(
                            _hip.hipMemUnmap(ctypes.c_void_p(mapped_va), size),
                            "hipMemUnmap",
                        ),
                    )
                )
            steps.append(
                (
                    "hipMemRelease",
                    lambda: _hip_try(
                        _hip.hipMemRelease(hipMemGenericAllocationHandle_t(imported_handle)),
                        "hipMemRelease",
                    ),
                )
            )
            if va_owned and mapped_va:
                steps.append(
                    (
                        "hipMemAddressFree",
                        lambda: _hip_try(
                            _hip.hipMemAddressFree(ctypes.c_void_p(mapped_va), size),
                            "hipMemAddressFree",
                        ),
                    )
                )
            _cleanup_after_failure(*steps)
            raise

        tag = "driver_va" if va_owned else "caller_va"
        return PeerMapping(
            peer_rank=peer_rank,
            transport=InterconnectLevel.INTRA_RACK_FABRIC,
            remote_va=mapped_va,
            size=size,
            _driver_handle=(tag, imported_handle),
        )

    def cleanup(self, target: LocalAllocation | PeerMapping) -> None:
        """Release a local HIP fabric allocation or imported HIP fabric mapping."""
        if isinstance(target, LocalAllocation):
            self._cleanup_local(target)
            return
        if isinstance(target, PeerMapping):
            self._cleanup_import(target)
            return
        raise AmdFabricError(f"Unsupported cleanup target: {type(target).__name__}")

    def _cleanup_import(self, mapping: PeerMapping) -> None:
        """Unmap, release, and conditionally free an imported HIP fabric mapping."""
        self._check_initialized()
        if isinstance(mapping._driver_handle, tuple) and len(mapping._driver_handle) == 2:
            tag, imported_handle = mapping._driver_handle
        else:
            tag = "driver_va"
            imported_handle = mapping._driver_handle

        steps: list[tuple[str, Callable[[], None]]] = [
            (
                "hipMemUnmap",
                lambda: _hip_try(
                    _hip.hipMemUnmap(ctypes.c_void_p(mapping.remote_va), mapping.size),
                    "hipMemUnmap",
                ),
            ),
            (
                "hipMemRelease",
                lambda: _hip_try(
                    _hip.hipMemRelease(hipMemGenericAllocationHandle_t(int(imported_handle))),
                    "hipMemRelease",
                ),
            ),
        ]
        if tag == "driver_va":
            steps.append(
                (
                    "hipMemAddressFree",
                    lambda: _hip_try(
                        _hip.hipMemAddressFree(ctypes.c_void_p(mapping.remote_va), mapping.size),
                        "hipMemAddressFree",
                    ),
                )
            )
        _run_cleanup_steps(*steps)

    def _cleanup_local(self, allocation: LocalAllocation) -> None:
        """Unmap, release, and conditionally free a local HIP fabric allocation."""
        self._check_initialized()
        steps: list[tuple[str, Callable[[], None]]] = [
            (
                "hipMemUnmap",
                lambda: _hip_try(
                    _hip.hipMemUnmap(ctypes.c_void_p(allocation.va), allocation.size),
                    "hipMemUnmap",
                ),
            ),
            (
                "hipMemRelease",
                lambda: _hip_try(
                    _hip.hipMemRelease(hipMemGenericAllocationHandle_t(int(allocation.handle))),
                    "hipMemRelease",
                ),
            ),
        ]
        if allocation._va_owned:
            steps.append(
                (
                    "hipMemAddressFree",
                    lambda: _hip_try(
                        _hip.hipMemAddressFree(ctypes.c_void_p(allocation.va), allocation.size),
                        "hipMemAddressFree",
                    ),
                )
            )
        _run_cleanup_steps(*steps)

    def get_minimum_granularity(self) -> int:
        """Return the HIP VMM allocation granularity for this device."""
        self._check_initialized()
        return self._get_granularity()

    def reserve_va(self, size: int, alignment: int = 0) -> int:
        """Reserve a HIP virtual address range without backing memory."""
        self._check_initialized()
        if alignment == 0:
            alignment = self._get_granularity()

        reserved = ctypes.c_void_p()
        _hip_try(
            _hip.hipMemAddressReserve(ctypes.byref(reserved), size, alignment, None, 0),
            "hipMemAddressReserve",
        )
        if reserved.value is None:
            raise AmdFabricError("hipMemAddressReserve returned a null VA")
        return int(reserved.value)

    def free_va(self, va: int, size: int) -> None:
        """Free a HIP VA range previously returned by reserve_va."""
        self._check_initialized()
        _hip_try(_hip.hipMemAddressFree(ctypes.c_void_p(va), size), "hipMemAddressFree")
