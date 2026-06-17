# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""NVIDIA CUDA driver-API local memory driver."""

from __future__ import annotations

import ctypes
import logging
import os
import struct
from collections.abc import Callable
from typing import Any, Optional

from iris.drivers.base import (
    BaseDriver,
    DriverError,
    DriverNotSupported,
    LocalAllocation,
    PeerMapping,
)
from iris.host.distributed.topology import InterconnectLevel

logger = logging.getLogger("iris.drivers.local.nvidia")

__all__ = [
    "LocalCudaError",
    "LocalCudaNotSupported",
    "LocalCudaDriver",
]

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

_CUDA_HANDLE_FMT = "=i"
_CUDA_HANDLE_BYTES = struct.calcsize(_CUDA_HANDLE_FMT)

_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3


class LocalCudaError(DriverError):
    """CUDA local VMM operation failed."""


class LocalCudaNotSupported(DriverNotSupported):
    """The local CUDA driver stack does not support this driver."""


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


def _get_required_cuda_symbol(name: str) -> Any:
    if _cuda_driver is None:
        raise LocalCudaNotSupported("CUDA driver library (libcuda.so) not found")

    symbol = getattr(_cuda_driver, name, None)
    if symbol is None:
        raise LocalCudaNotSupported(f"CUDA driver missing required VMM symbol: {name}")
    return symbol


def _configure_signatures() -> None:
    """Configure ctypes signatures for all CUDA driver functions used here."""
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


def _cuda_try(err: int, op_name: str = "CUDA operation") -> None:
    """Check a CUDA driver return code and raise a driver exception on error."""
    if err == CUDA_SUCCESS:
        return

    error_name = str(err)
    if _cuda_driver is not None and hasattr(_cuda_driver, "cuGetErrorName"):
        ptr = ctypes.c_char_p()
        if _cuda_driver.cuGetErrorName(err, ctypes.byref(ptr)) == CUDA_SUCCESS and ptr.value:
            error_name = ptr.value.decode("utf-8")

    message = f"{op_name} failed with {error_name} ({err})"
    if err == CUDA_ERROR_NOT_SUPPORTED:
        raise LocalCudaNotSupported(message)
    raise LocalCudaError(message)


def _round_up(value: int, granularity: int) -> int:
    if granularity <= 0:
        raise ValueError(f"granularity must be > 0, got {granularity}")
    return ((value + granularity - 1) // granularity) * granularity


def _normalize_handle_bytes(raw_handle: bytes) -> bytes:
    if isinstance(raw_handle, memoryview):
        data = raw_handle.tobytes()
    elif isinstance(raw_handle, (bytes, bytearray)):
        data = bytes(raw_handle)
    else:
        try:
            data = bytes(raw_handle)
        except Exception as exc:
            raise LocalCudaError("Unable to convert POSIX-FD handle object to bytes") from exc

    if len(data) != _CUDA_HANDLE_BYTES:
        raise LocalCudaError(f"CUDA POSIX-FD handle must be {_CUDA_HANDLE_BYTES} bytes, got {len(data)}")
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


class LocalCudaDriver(BaseDriver):
    """
    NVIDIA CUDA driver-API VMM local driver.

    This driver uses libcuda.so, not the CUDA runtime API. Exported handles are
    POSIX file descriptors encoded as bytes; the caller is responsible for
    delivering the FD across processes, for example with SCM_RIGHTS. POSIX-FD
    handles require source and destination processes to share the same OS
    namespace, typically on the same machine.
    """

    def __init__(self) -> None:
        self._device_ordinal: int = 0
        self._granularity: Optional[int] = None
        self._initialized: bool = False

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise LocalCudaError("LocalCudaDriver not initialized - call initialize() first")

    def _make_alloc_props(self) -> _MemAllocationProp:
        props = _MemAllocationProp()
        props.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        props.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
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
        _cuda_try(
            _cuda_driver.cuMemSetAccess(va, size, ctypes.byref(desc), 1),
            "cuMemSetAccess",
        )

    def initialize(self, device_ordinal: int) -> None:
        """Prepare the CUDA driver context and bind this instance to one GPU."""
        if _cuda_driver is None:
            raise LocalCudaNotSupported("CUDA driver library (libcuda.so) not found")

        _configure_signatures()
        _cuda_try(_cuda_driver.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        _cuda_try(_cuda_driver.cuDeviceGet(ctypes.byref(dev), device_ordinal), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        _cuda_try(
            _cuda_driver.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value),
            "cuDevicePrimaryCtxRetain",
        )
        _cuda_try(_cuda_driver.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self._device_ordinal = device_ordinal
        self._granularity = None
        self._initialized = True
        logger.info("LocalCudaDriver initialized (device %d)", device_ordinal)

    def allocate_exportable(
        self,
        size: int,
        va: Optional[int] = None,
        *,
        access_va: Optional[int] = None,
        access_size: Optional[int] = None,
    ) -> LocalAllocation:
        """
        Allocate CUDA VMM memory exportable as a POSIX FD.

        If va is supplied, the caller must already own a sufficiently large,
        granularity-aligned VA range containing [va, va + size).
        """
        self._check_initialized()
        if (access_va is None) != (access_size is None):
            raise LocalCudaError("access_va and access_size must be provided together")
        props = self._make_alloc_props()
        granularity = self._get_granularity()
        alloc_size = _round_up(size, granularity)

        reserved_va = va is None
        mapped_va = int(va) if va is not None else 0
        handle = ctypes.c_uint64()
        mapped = False

        try:
            if reserved_va:
                reserved = ctypes.c_uint64()
                _cuda_try(
                    _cuda_driver.cuMemAddressReserve(ctypes.byref(reserved), alloc_size, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
                mapped_va = int(reserved.value)
            _cuda_try(
                _cuda_driver.cuMemCreate(ctypes.byref(handle), alloc_size, ctypes.byref(props), 0),
                "cuMemCreate",
            )
            _cuda_try(
                _cuda_driver.cuMemMap(mapped_va, alloc_size, 0, handle.value, 0),
                "cuMemMap",
            )
            mapped = True
            self._mem_set_access(
                int(access_va) if access_va is not None else mapped_va,
                int(access_size) if access_size is not None else alloc_size,
            )
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
                        "cuMemUnmap",
                        lambda: _cuda_try(_cuda_driver.cuMemUnmap(mapped_va, alloc_size), "cuMemUnmap"),
                    )
                )
            if handle.value:
                steps.append(
                    (
                        "cuMemRelease",
                        lambda: _cuda_try(_cuda_driver.cuMemRelease(handle.value), "cuMemRelease"),
                    )
                )
            if reserved_va and mapped_va:
                steps.append(
                    (
                        "cuMemAddressFree",
                        lambda: _cuda_try(
                            _cuda_driver.cuMemAddressFree(mapped_va, alloc_size),
                            "cuMemAddressFree",
                        ),
                    )
                )
            _cleanup_after_failure(*steps)
            raise

    def export_handle(self, allocation: LocalAllocation) -> bytes:
        """Export a 4-byte native-endian POSIX-FD descriptor for a local allocation."""
        self._check_initialized()
        fd = ctypes.c_int(-1)
        _cuda_try(
            _cuda_driver.cuMemExportToShareableHandle(
                ctypes.byref(fd),
                int(allocation.handle),
                _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            ),
            "cuMemExportToShareableHandle",
        )
        return struct.pack(_CUDA_HANDLE_FMT, int(fd.value))

    def _import_handle(self, handle_bytes: bytes) -> int:
        handle_bytes = _normalize_handle_bytes(handle_bytes)
        fd_value = struct.unpack(_CUDA_HANDLE_FMT, handle_bytes)[0]
        imported = ctypes.c_uint64()
        err = _cuda_driver.cuMemImportFromShareableHandle(
            ctypes.byref(imported),
            ctypes.c_void_p(fd_value),
            _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
        if err != CUDA_SUCCESS:
            try:
                os.close(fd_value)
            except OSError:
                pass
            _cuda_try(err, "cuMemImportFromShareableHandle")
        os.close(fd_value)
        return int(imported.value)

    def import_and_map(
        self,
        peer_rank: int,
        handle_bytes: bytes,
        size: int,
        va: Optional[int] = None,
        *,
        access_va: Optional[int] = None,
        access_size: Optional[int] = None,
    ) -> PeerMapping:
        """Import a POSIX-FD handle and map it into local CUDA VMM VA space."""
        self._check_initialized()
        if (access_va is None) != (access_size is None):
            raise LocalCudaError("access_va and access_size must be provided together")
        imported_handle = self._import_handle(handle_bytes)

        granularity = self._get_granularity()
        va_owned = va is None
        mapped_va = int(va) if va is not None else 0
        mapped = False
        try:
            if va_owned:
                reserved = ctypes.c_uint64()
                _cuda_try(
                    _cuda_driver.cuMemAddressReserve(ctypes.byref(reserved), size, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
                mapped_va = int(reserved.value)
            _cuda_try(
                _cuda_driver.cuMemMap(mapped_va, size, 0, imported_handle, 0),
                "cuMemMap",
            )
            mapped = True
            self._mem_set_access(
                int(access_va) if access_va is not None else mapped_va,
                int(access_size) if access_size is not None else size,
            )
        except Exception:
            steps: list[tuple[str, Callable[[], None]]] = []
            if mapped:
                steps.append(
                    (
                        "cuMemUnmap",
                        lambda: _cuda_try(_cuda_driver.cuMemUnmap(mapped_va, size), "cuMemUnmap"),
                    )
                )
            steps.append(
                (
                    "cuMemRelease",
                    lambda: _cuda_try(_cuda_driver.cuMemRelease(imported_handle), "cuMemRelease"),
                )
            )
            if va_owned and mapped_va:
                steps.append(
                    (
                        "cuMemAddressFree",
                        lambda: _cuda_try(
                            _cuda_driver.cuMemAddressFree(mapped_va, size),
                            "cuMemAddressFree",
                        ),
                    )
                )
            _cleanup_after_failure(*steps)
            raise

        tag = "driver_va" if va_owned else "caller_va"
        return PeerMapping(
            peer_rank=peer_rank,
            transport=InterconnectLevel.INTRA_NODE,
            remote_va=mapped_va,
            size=size,
            _driver_handle=(tag, imported_handle),
        )

    def cleanup_import(self, mapping: PeerMapping) -> None:
        """Unmap, release, and free an imported CUDA VMM mapping."""
        self._check_initialized()
        if isinstance(mapping._driver_handle, tuple) and len(mapping._driver_handle) == 2:
            tag, imported_handle = mapping._driver_handle
        else:
            tag = "driver_va"
            imported_handle = mapping._driver_handle

        steps: list[tuple[str, Callable[[], None]]] = [
            (
                "cuMemUnmap",
                lambda: _cuda_try(
                    _cuda_driver.cuMemUnmap(mapping.remote_va, mapping.size),
                    "cuMemUnmap",
                ),
            ),
            (
                "cuMemRelease",
                lambda: _cuda_try(_cuda_driver.cuMemRelease(imported_handle), "cuMemRelease"),
            ),
        ]
        if tag == "driver_va":
            steps.append(
                (
                    "cuMemAddressFree",
                    lambda: _cuda_try(
                        _cuda_driver.cuMemAddressFree(mapping.remote_va, mapping.size),
                        "cuMemAddressFree",
                    ),
                )
            )
        _run_cleanup_steps(*steps)

    def cleanup_local(self, allocation: LocalAllocation) -> None:
        """Unmap, release, and conditionally free a local CUDA VMM allocation."""
        self._check_initialized()
        steps = [
            (
                "cuMemUnmap",
                lambda: _cuda_try(
                    _cuda_driver.cuMemUnmap(allocation.va, allocation.size),
                    "cuMemUnmap",
                ),
            ),
            (
                "cuMemRelease",
                lambda: _cuda_try(_cuda_driver.cuMemRelease(allocation.handle), "cuMemRelease"),
            ),
        ]
        if allocation._va_owned:
            steps.append(
                (
                    "cuMemAddressFree",
                    lambda: _cuda_try(
                        _cuda_driver.cuMemAddressFree(allocation.va, allocation.size),
                        "cuMemAddressFree",
                    ),
                )
            )
        _run_cleanup_steps(*steps)

    def get_minimum_granularity(self) -> int:
        """Return the CUDA VMM allocation granularity for this device."""
        self._check_initialized()
        return self._get_granularity()

    def reserve_va(self, size: int, alignment: int = 0) -> int:
        """Reserve a CUDA virtual address range without backing memory."""
        self._check_initialized()
        if alignment == 0:
            alignment = self._get_granularity()

        reserved = ctypes.c_uint64()
        _cuda_try(
            _cuda_driver.cuMemAddressReserve(ctypes.byref(reserved), size, alignment, 0, 0),
            "cuMemAddressReserve",
        )
        return int(reserved.value)

    def free_va(self, va: int, size: int) -> None:
        """Free a CUDA VA range previously returned by reserve_va."""
        self._check_initialized()
        _cuda_try(_cuda_driver.cuMemAddressFree(va, size), "cuMemAddressFree")
