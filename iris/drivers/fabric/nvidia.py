# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
NVIDIA CUDA VMM fabric driver.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Optional

import torch

from iris.drivers.base import (
    BaseFabricDriver,
    DriverError,
    DriverNotSupported,
    LocalAllocation,
    PeerMapping,
)
from iris.topology import InterconnectLevel

logger = logging.getLogger("iris.drivers.fabric")

__all__ = [
    "CudaFabricError",
    "CudaFabricNotSupported",
    "FABRIC_HANDLE_BYTES",
    "NvidiaFabricDriver",
]

# Load CUDA driver library at module level
_cuda_driver = None
try:
    _cuda_driver = ctypes.CDLL("libcuda.so.1")
except OSError:
    try:
        _cuda_driver = ctypes.CDLL("libcuda.so")
    except OSError:
        pass

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_SUPPORTED = 801
FABRIC_HANDLE_BYTES = 64

# CUDA VMM constants
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_HANDLE_TYPE_FABRIC = 0x8
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3


class CudaFabricError(DriverError):
    """CUDA fabric/VMM operation failed."""


class CudaFabricNotSupported(DriverNotSupported):
    """The local CUDA stack does not support fabric handles."""


def _cuda_try(err: int, op_name: str = "CUDA operation") -> None:
    """Check CUDA driver return code and raise on error."""
    if err == CUDA_SUCCESS:
        return
    error_name = str(err)
    if _cuda_driver is not None and hasattr(_cuda_driver, "cuGetErrorName"):
        ptr = ctypes.c_char_p()
        if _cuda_driver.cuGetErrorName(err, ctypes.byref(ptr)) == CUDA_SUCCESS and ptr.value:
            error_name = ptr.value.decode("utf-8")
    message = f"{op_name} failed with {error_name} ({err})"
    if err == CUDA_ERROR_NOT_SUPPORTED:
        raise CudaFabricNotSupported(message)
    raise CudaFabricError(message)


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
                raise CudaFabricError("Unable to convert fabric handle object to bytes") from exc

    if len(data) != FABRIC_HANDLE_BYTES:
        raise CudaFabricError(f"Fabric handle serialization expected {FABRIC_HANDLE_BYTES} bytes, got {len(data)}")
    return data


def _get_required_cuda_symbol(name: str) -> Any:
    if _cuda_driver is None:
        raise CudaFabricNotSupported("CUDA driver library (libcuda.so) not found")

    symbol = getattr(_cuda_driver, name, None)
    if symbol is None:
        raise CudaFabricNotSupported(f"CUDA driver missing required VMM symbol: {name}")
    return symbol


def _run_cleanup_steps(*steps) -> None:
    first_error = None
    for step in steps:
        try:
            step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


# ctypes structure definitions for CUDA VMM API
class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemAllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _MemAllocationFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_ulonglong)]


def _configure_cuda_signatures() -> None:
    """Configure ctypes signatures for the required CUDA VMM driver API."""
    if _cuda_driver is None:
        return

    cu_init = _get_required_cuda_symbol("cuInit")
    cu_device_get = _get_required_cuda_symbol("cuDeviceGet")
    cu_device_primary_ctx_retain = _get_required_cuda_symbol("cuDevicePrimaryCtxRetain")
    cu_ctx_set_current = _get_required_cuda_symbol("cuCtxSetCurrent")
    cu_mem_get_allocation_granularity = _get_required_cuda_symbol("cuMemGetAllocationGranularity")
    cu_mem_address_reserve = _get_required_cuda_symbol("cuMemAddressReserve")
    cu_mem_address_free = _get_required_cuda_symbol("cuMemAddressFree")
    cu_mem_create = _get_required_cuda_symbol("cuMemCreate")
    cu_mem_release = _get_required_cuda_symbol("cuMemRelease")
    cu_mem_map = _get_required_cuda_symbol("cuMemMap")
    cu_mem_unmap = _get_required_cuda_symbol("cuMemUnmap")
    cu_mem_set_access = _get_required_cuda_symbol("cuMemSetAccess")
    cu_mem_export_to_shareable_handle = _get_required_cuda_symbol("cuMemExportToShareableHandle")
    cu_mem_import_from_shareable_handle = _get_required_cuda_symbol("cuMemImportFromShareableHandle")

    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int

    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_device_get.restype = ctypes.c_int

    cu_device_primary_ctx_retain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    cu_device_primary_ctx_retain.restype = ctypes.c_int

    cu_ctx_set_current.argtypes = [ctypes.c_void_p]
    cu_ctx_set_current.restype = ctypes.c_int

    cu_mem_get_allocation_granularity.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(_MemAllocationProp),
        ctypes.c_int,
    ]
    cu_mem_get_allocation_granularity.restype = ctypes.c_int

    cu_mem_address_reserve.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_ulonglong,
    ]
    cu_mem_address_reserve.restype = ctypes.c_int

    cu_mem_address_free.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
    cu_mem_address_free.restype = ctypes.c_int

    cu_mem_create.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(_MemAllocationProp),
        ctypes.c_ulonglong,
    ]
    cu_mem_create.restype = ctypes.c_int

    cu_mem_release.argtypes = [ctypes.c_uint64]
    cu_mem_release.restype = ctypes.c_int

    cu_mem_map.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_ulonglong,
    ]
    cu_mem_map.restype = ctypes.c_int

    cu_mem_unmap.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
    cu_mem_unmap.restype = ctypes.c_int

    cu_mem_set_access.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(_MemAccessDesc),
        ctypes.c_size_t,
    ]
    cu_mem_set_access.restype = ctypes.c_int

    cu_mem_export_to_shareable_handle.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_ulonglong,
    ]
    cu_mem_export_to_shareable_handle.restype = ctypes.c_int

    cu_mem_import_from_shareable_handle.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    cu_mem_import_from_shareable_handle.restype = ctypes.c_int

    cu_get_error_name = getattr(_cuda_driver, "cuGetErrorName", None)
    if cu_get_error_name is not None:
        cu_get_error_name.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        cu_get_error_name.restype = ctypes.c_int


class NvidiaFabricDriver(BaseFabricDriver):
    """
    NVIDIA CUDA VMM fabric driver.

    Uses ctypes to interface with libcuda.so for CUDA Virtual Memory Management
    operations required for fabric handle export/import.
    """

    def __init__(self) -> None:
        self._device_ordinal: int = 0
        self._granularity: Optional[int] = None
        self._initialized: bool = False

    def _make_alloc_props(self) -> _MemAllocationProp:
        props = _MemAllocationProp()
        props.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        props.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_FABRIC
        props.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        props.location.id = self._device_ordinal
        props.win32HandleMetaData = None
        return props

    def _get_granularity(self) -> int:
        if self._granularity is not None:
            return self._granularity
        props = self._make_alloc_props()
        granularity = ctypes.c_size_t()
        _cuda_try(
            _cuda_driver.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(props),
                _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
        self._granularity = int(granularity.value)
        return self._granularity

    def _mem_set_access(self, va: int, size: int) -> None:
        desc = _MemAccessDesc()
        desc.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self._device_ordinal
        desc.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _cuda_try(_cuda_driver.cuMemSetAccess(va, size, ctypes.byref(desc), 1), "cuMemSetAccess")

    def initialize(self, device_ordinal: int) -> None:
        if _cuda_driver is None:
            raise CudaFabricNotSupported("CUDA driver library (libcuda.so) not found")

        _configure_cuda_signatures()
        _cuda_try(_cuda_driver.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        _cuda_try(_cuda_driver.cuDeviceGet(ctypes.byref(dev), device_ordinal), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        _cuda_try(_cuda_driver.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value), "cuDevicePrimaryCtxRetain")
        _cuda_try(_cuda_driver.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self._device_ordinal = device_ordinal
        self._granularity = None
        self._initialized = True
        logger.info("NvidiaFabricDriver initialized (device %d)", device_ordinal)

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise CudaFabricError("NvidiaFabricDriver not initialized — call initialize() first")

    def allocate_exportable(self, size: int) -> LocalAllocation:
        self._check_initialized()
        props = self._make_alloc_props()
        granularity = self._get_granularity()
        alloc_size = _round_up(size, granularity)

        va = ctypes.c_uint64()
        handle = ctypes.c_uint64()
        mapped = False

        try:
            _cuda_try(
                _cuda_driver.cuMemAddressReserve(ctypes.byref(va), alloc_size, granularity, 0, 0),
                "cuMemAddressReserve",
            )
            _cuda_try(
                _cuda_driver.cuMemCreate(ctypes.byref(handle), alloc_size, ctypes.byref(props), 0),
                "cuMemCreate",
            )
            _cuda_try(_cuda_driver.cuMemMap(va.value, alloc_size, 0, handle.value, 0), "cuMemMap")
            mapped = True
            self._mem_set_access(int(va.value), alloc_size)
            return LocalAllocation(va=int(va.value), size=alloc_size, handle=int(handle.value))
        except Exception:
            if mapped:
                try:
                    _cuda_try(_cuda_driver.cuMemUnmap(va.value, alloc_size), "cuMemUnmap")
                except Exception:
                    pass
            if handle.value:
                try:
                    _cuda_try(_cuda_driver.cuMemRelease(handle.value), "cuMemRelease")
                except Exception:
                    pass
            if va.value:
                try:
                    _cuda_try(_cuda_driver.cuMemAddressFree(va.value, alloc_size), "cuMemAddressFree")
                except Exception:
                    pass
            raise

    def export_handle(self, allocation: LocalAllocation) -> bytes:
        self._check_initialized()
        raw = (ctypes.c_ubyte * FABRIC_HANDLE_BYTES)()
        _cuda_try(
            _cuda_driver.cuMemExportToShareableHandle(
                ctypes.byref(raw),
                int(allocation.handle),
                _CU_MEM_HANDLE_TYPE_FABRIC,
                0,
            ),
            "cuMemExportToShareableHandle",
        )
        return bytes(raw)

    def _import_handle(self, handle_bytes: bytes) -> int:
        handle_bytes = _normalize_fabric_handle_bytes(handle_bytes)
        imported = ctypes.c_uint64()
        raw = (ctypes.c_ubyte * FABRIC_HANDLE_BYTES).from_buffer_copy(handle_bytes)
        _cuda_try(
            _cuda_driver.cuMemImportFromShareableHandle(
                ctypes.byref(imported),
                ctypes.byref(raw),
                _CU_MEM_HANDLE_TYPE_FABRIC,
            ),
            "cuMemImportFromShareableHandle",
        )
        return int(imported.value)

    def import_and_map(self, peer_rank: int, handle_bytes: bytes, size: int) -> PeerMapping:
        self._check_initialized()
        imported_handle = self._import_handle(handle_bytes)

        granularity = self._get_granularity()
        va = ctypes.c_uint64()

        mapped = False
        try:
            _cuda_try(
                _cuda_driver.cuMemAddressReserve(ctypes.byref(va), size, granularity, 0, 0),
                "cuMemAddressReserve",
            )
            _cuda_try(_cuda_driver.cuMemMap(va.value, size, 0, imported_handle, 0), "cuMemMap")
            mapped = True
            self._mem_set_access(int(va.value), size)
        except Exception:
            if mapped:
                try:
                    _cuda_try(_cuda_driver.cuMemUnmap(va.value, size), "cuMemUnmap")
                except Exception:
                    pass
            try:
                _cuda_try(_cuda_driver.cuMemRelease(imported_handle), "cuMemRelease")
            except Exception:
                pass
            if va.value:
                try:
                    _cuda_try(_cuda_driver.cuMemAddressFree(va.value, size), "cuMemAddressFree")
                except Exception:
                    pass
            raise

        return PeerMapping(
            peer_rank=peer_rank,
            transport=InterconnectLevel.INTRA_RACK_FABRIC,
            remote_va=int(va.value),
            size=size,
            _driver_handle=imported_handle,
        )

    def cleanup_import(self, mapping: PeerMapping) -> None:
        self._check_initialized()
        _run_cleanup_steps(
            lambda: _cuda_try(_cuda_driver.cuMemUnmap(mapping.remote_va, mapping.size), "cuMemUnmap"),
            lambda: _cuda_try(_cuda_driver.cuMemRelease(mapping._driver_handle), "cuMemRelease"),
            lambda: _cuda_try(_cuda_driver.cuMemAddressFree(mapping.remote_va, mapping.size), "cuMemAddressFree"),
        )

    def cleanup_local(self, allocation: LocalAllocation) -> None:
        self._check_initialized()
        _run_cleanup_steps(
            lambda: _cuda_try(_cuda_driver.cuMemUnmap(allocation.va, allocation.size), "cuMemUnmap"),
            lambda: _cuda_try(_cuda_driver.cuMemRelease(allocation.handle), "cuMemRelease"),
            lambda: _cuda_try(_cuda_driver.cuMemAddressFree(allocation.va, allocation.size), "cuMemAddressFree"),
        )
