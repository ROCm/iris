# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""AMD HIP local memory driver."""

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
    ExportableMemory,
    LocalAllocation,
    MappingPlacement,
    PeerMapping,
)
from iris.host.distributed.topology import InterconnectLevel

logger = logging.getLogger("iris.drivers.local.amd")

__all__ = [
    "LocalHipError",
    "LocalHipNotSupported",
    "LocalHipDriver",
]

_hip = None
try:
    _hip = ctypes.cdll.LoadLibrary("libamdhip64.so")
except OSError:
    pass

HIP_SUCCESS = 0
HIP_ERROR_NOT_SUPPORTED = 801

hipMemAllocationTypePinned = 0x1
hipMemHandleTypePosixFileDescriptor = 0x1
hipMemLocationTypeDevice = 0x1
hipMemAllocationGranularityRecommended = 0x1
hipMemAccessFlagsProtReadWrite = 0x3
hipExternalMemoryHandleTypeOpaqueFd = 1

hipMemGenericAllocationHandle_t = ctypes.c_void_p
hipExternalMemory_t = ctypes.c_void_p

_AMD_HANDLE_FMT = "=iQQ"
_AMD_HANDLE_BYTES = struct.calcsize(_AMD_HANDLE_FMT)


class LocalHipError(DriverError):
    """HIP local-memory operation failed."""


class LocalHipNotSupported(DriverNotSupported):
    """The local HIP stack does not support this driver."""


class hipMemLocation(ctypes.Structure):
    """Structure describing a HIP memory location."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class hipMemAllocationProp(ctypes.Structure):
    """Properties for a HIP VMem allocation."""

    class _allocFlags(ctypes.Structure):
        _fields_ = [
            ("smc", ctypes.c_ubyte),
            ("l2", ctypes.c_ubyte),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", hipMemLocation),
        ("win32Handle", ctypes.c_void_p),
        ("allocFlags", _allocFlags),
    ]


class hipMemAccessDesc(ctypes.Structure):
    """Access descriptor for a HIP VMem mapping."""

    _fields_ = [
        ("location", hipMemLocation),
        ("flags", ctypes.c_int),
    ]


class hipExternalMemoryHandleDesc(ctypes.Structure):
    """Descriptor for importing HIP external memory from a DMA-BUF FD."""

    class HandleUnion(ctypes.Union):
        _fields_ = [
            ("fd", ctypes.c_int),
            ("win32", ctypes.c_void_p * 2),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("_pad", ctypes.c_int),
        ("handle", HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("_pad2", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class hipExternalMemoryBufferDesc(ctypes.Structure):
    """Descriptor for mapping an imported HIP external-memory buffer."""

    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


def _get_required_hip_symbol(name: str) -> Any:
    if _hip is None:
        raise LocalHipNotSupported("libamdhip64.so not found")

    symbol = getattr(_hip, name, None)
    if symbol is None:
        raise LocalHipNotSupported(f"HIP runtime missing required symbol: {name}")
    return symbol


def _configure_signatures() -> None:
    """Configure ctypes signatures for all HIP functions used by this driver."""
    if _hip is None:
        return

    hip_set_device = _get_required_hip_symbol("hipSetDevice")
    hip_mem_get_allocation_granularity = _get_required_hip_symbol("hipMemGetAllocationGranularity")
    hip_mem_create = _get_required_hip_symbol("hipMemCreate")
    hip_mem_address_reserve = _get_required_hip_symbol("hipMemAddressReserve")
    hip_mem_map = _get_required_hip_symbol("hipMemMap")
    hip_mem_set_access = _get_required_hip_symbol("hipMemSetAccess")
    hip_mem_unmap = _get_required_hip_symbol("hipMemUnmap")
    hip_mem_release = _get_required_hip_symbol("hipMemRelease")
    hip_mem_address_free = _get_required_hip_symbol("hipMemAddressFree")
    hip_mem_get_address_range = _get_required_hip_symbol("hipMemGetAddressRange")
    hip_mem_get_handle_for_address_range = _get_required_hip_symbol("hipMemGetHandleForAddressRange")
    hip_mem_import_from_shareable_handle = _get_required_hip_symbol("hipMemImportFromShareableHandle")
    hip_import_external_memory = _get_required_hip_symbol("hipImportExternalMemory")
    hip_external_memory_get_mapped_buffer = _get_required_hip_symbol("hipExternalMemoryGetMappedBuffer")
    hip_destroy_external_memory = _get_required_hip_symbol("hipDestroyExternalMemory")
    hip_get_error_string = _get_required_hip_symbol("hipGetErrorString")

    hip_set_device.argtypes = [ctypes.c_int]
    hip_set_device.restype = ctypes.c_int

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

    hip_mem_get_address_range.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    hip_mem_get_address_range.restype = ctypes.c_int

    hip_mem_get_handle_for_address_range.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_ulonglong,
    ]
    hip_mem_get_handle_for_address_range.restype = ctypes.c_int

    hip_mem_import_from_shareable_handle.argtypes = [
        ctypes.POINTER(hipMemGenericAllocationHandle_t),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    hip_mem_import_from_shareable_handle.restype = ctypes.c_int

    hip_import_external_memory.argtypes = [
        ctypes.POINTER(hipExternalMemory_t),
        ctypes.POINTER(hipExternalMemoryHandleDesc),
    ]
    hip_import_external_memory.restype = ctypes.c_int

    hip_external_memory_get_mapped_buffer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipExternalMemory_t,
        ctypes.POINTER(hipExternalMemoryBufferDesc),
    ]
    hip_external_memory_get_mapped_buffer.restype = ctypes.c_int

    hip_destroy_external_memory.argtypes = [hipExternalMemory_t]
    hip_destroy_external_memory.restype = ctypes.c_int

    hip_get_error_string.argtypes = [ctypes.c_int]
    hip_get_error_string.restype = ctypes.c_char_p


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
        raise LocalHipNotSupported(message)
    raise LocalHipError(message)


def _round_up(value: int, granularity: int) -> int:
    if granularity <= 0:
        raise ValueError(f"granularity must be > 0, got {granularity}")
    return ((value + granularity - 1) // granularity) * granularity


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


class LocalHipDriver(BaseDriver):
    """
    AMD HIP VMem local driver using DMA-BUF handles for peer import/export.

    hipSetDevice is per-thread; use each driver instance from the thread that
    called initialize().
    """

    def __init__(self) -> None:
        self._device_ordinal: int = 0
        self._granularity: Optional[int] = None
        self._initialized: bool = False

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise LocalHipError("LocalHipDriver not initialized - call initialize() first")

    def _make_alloc_props(self) -> hipMemAllocationProp:
        props = hipMemAllocationProp()
        props.type = hipMemAllocationTypePinned
        props.requestedHandleType = hipMemHandleTypePosixFileDescriptor
        props.location.type = hipMemLocationTypeDevice
        props.location.id = self._device_ordinal
        props.win32Handle = None
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

    def initialize(self, device_ordinal: int) -> None:
        """Prepare the HIP runtime and bind this driver instance to one GPU."""
        if _hip is None:
            raise LocalHipNotSupported("libamdhip64.so not found")

        _configure_signatures()
        _hip_try(_hip.hipSetDevice(device_ordinal), "hipSetDevice")
        self._device_ordinal = device_ordinal
        self._granularity = None
        self._initialized = True
        logger.info("LocalHipDriver initialized (device %d)", device_ordinal)

    def allocate_exportable(
        self,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> LocalAllocation:
        """
        Allocate HIP VMem exportable as a DMA-BUF.

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
                            _hip.hipMemRelease(hipMemGenericAllocationHandle_t(handle.value)),
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

    def _export_range(self, va: int, size: int) -> bytes:
        base_ptr = ctypes.c_void_p()
        base_size = ctypes.c_size_t()
        allocation_ptr = ctypes.c_void_p(int(va))
        _hip_try(
            _hip.hipMemGetAddressRange(ctypes.byref(base_ptr), ctypes.byref(base_size), allocation_ptr),
            "hipMemGetAddressRange",
        )
        if base_ptr.value is None:
            raise LocalHipError("hipMemGetAddressRange returned a null base pointer")

        fd = ctypes.c_int(-1)
        _hip_try(
            _hip.hipMemGetHandleForAddressRange(ctypes.byref(fd), allocation_ptr, size, 1, 0),
            "hipMemGetHandleForAddressRange",
        )
        fd_value = int(fd.value)

        try:
            base_va = int(base_ptr.value)
            base_size_value = int(base_size.value)
            offset = int(va) - base_va
            if offset < 0 or offset + size > base_size_value:
                raise LocalHipError(
                    f"Allocation range va={va} size={size} exceeds base range va={base_va} size={base_size_value}"
                )

            return struct.pack(_AMD_HANDLE_FMT, fd_value, offset, base_size_value)
        except Exception:
            try:
                os.close(fd_value)
            except OSError:
                pass
            raise

    def export_handle(self, memory: ExportableMemory) -> bytes:
        """Export a 20-byte DMA-BUF descriptor for a local HIP memory range."""
        self._check_initialized()
        return self._export_range(memory.va, memory.size)

    def import_and_map(
        self,
        peer_rank: int,
        handle_bytes: bytes,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> PeerMapping:
        """Import a DMA-BUF descriptor and map it into local GPU address space."""
        self._check_initialized()
        if len(handle_bytes) != _AMD_HANDLE_BYTES:
            raise LocalHipError(f"AMD local handle must be {_AMD_HANDLE_BYTES} bytes, got {len(handle_bytes)}")

        fd, offset, base_size = struct.unpack(_AMD_HANDLE_FMT, handle_bytes)
        if size > base_size - offset:
            raise LocalHipError(f"Requested map size {size} exceeds imported base range {base_size} at offset {offset}")

        if placement is not None:
            mapped_va = int(placement.va)
            imported_handle = hipMemGenericAllocationHandle_t()
            mapped = False
            fd_open = True
            try:
                _hip_try(
                    _hip.hipMemImportFromShareableHandle(
                        ctypes.byref(imported_handle),
                        ctypes.c_void_p(fd),
                        hipMemHandleTypePosixFileDescriptor,
                    ),
                    "hipMemImportFromShareableHandle",
                )
                os.close(fd)
                fd_open = False

                _hip_try(
                    _hip.hipMemMap(ctypes.c_void_p(mapped_va), size, offset, imported_handle, 0),
                    "hipMemMap",
                )
                mapped = True
                access_base, access_bytes = placement.access_range(size, cumulative=True)
                self._mem_set_access(access_base, access_bytes)
                return PeerMapping(
                    peer_rank=peer_rank,
                    transport=InterconnectLevel.INTRA_NODE,
                    remote_va=mapped_va,
                    size=size,
                    _driver_handle=("vmm", int(imported_handle.value)),
                )
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
                if imported_handle.value:
                    steps.append(
                        (
                            "hipMemRelease",
                            lambda: _hip_try(
                                _hip.hipMemRelease(imported_handle),
                                "hipMemRelease",
                            ),
                        )
                    )
                if fd_open:
                    steps.append(("os.close", lambda: os.close(fd)))
                _cleanup_after_failure(*steps)
                raise

        mem_handle_desc = hipExternalMemoryHandleDesc()
        mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd
        mem_handle_desc.handle.fd = fd
        mem_handle_desc.size = base_size
        mem_handle_desc.flags = 0

        ext_mem = hipExternalMemory_t()
        try:
            # ROCm 7.1+ external memory import is preferred over
            # hipMemImportFromShareableHandle to avoid the ROCm 7.0 MemObjMap
            # segfault path for imported memory objects.
            err = _hip.hipImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(mem_handle_desc))
            if err != HIP_SUCCESS:
                try:
                    os.close(fd)
                except OSError:
                    pass
                _hip_try(err, "hipImportExternalMemory")

            buffer_desc = hipExternalMemoryBufferDesc()
            buffer_desc.offset = 0
            buffer_desc.size = base_size
            buffer_desc.flags = 0

            mapped_base = ctypes.c_void_p()
            _hip_try(
                _hip.hipExternalMemoryGetMappedBuffer(ctypes.byref(mapped_base), ext_mem, ctypes.byref(buffer_desc)),
                "hipExternalMemoryGetMappedBuffer",
            )
            if mapped_base.value is None:
                raise LocalHipError("hipExternalMemoryGetMappedBuffer returned a null pointer")

            remote_va = int(mapped_base.value) + int(offset)
            return PeerMapping(
                peer_rank=peer_rank,
                transport=InterconnectLevel.INTRA_NODE,
                remote_va=remote_va,
                size=size,
                _driver_handle=(ext_mem, base_size),
            )
        except Exception:
            if ext_mem.value:
                _cleanup_after_failure(
                    (
                        "hipDestroyExternalMemory",
                        lambda: _hip_try(
                            _hip.hipDestroyExternalMemory(ext_mem),
                            "hipDestroyExternalMemory",
                        ),
                    )
                )
            raise

    def cleanup(self, target: LocalAllocation | PeerMapping) -> None:
        """Release a local HIP allocation or imported HIP mapping."""
        if isinstance(target, LocalAllocation):
            self._cleanup_local(target)
            return
        if isinstance(target, PeerMapping):
            self._cleanup_import(target)
            return
        raise LocalHipError(f"Unsupported cleanup target: {type(target).__name__}")

    def _cleanup_import(self, mapping: PeerMapping) -> None:
        """Release an imported HIP external-memory mapping."""
        self._check_initialized()
        if (
            isinstance(mapping._driver_handle, tuple)
            and len(mapping._driver_handle) == 2
            and mapping._driver_handle[0] == "vmm"
        ):
            imported_handle = hipMemGenericAllocationHandle_t(mapping._driver_handle[1])
            _run_cleanup_steps(
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
                        _hip.hipMemRelease(imported_handle),
                        "hipMemRelease",
                    ),
                ),
            )
            return

        ext_mem, _base_size = mapping._driver_handle
        try:
            _hip_try(_hip.hipDestroyExternalMemory(ext_mem), "hipDestroyExternalMemory")
        except Exception:
            logger.warning("hipDestroyExternalMemory failed during import cleanup", exc_info=True)
            raise

    def _cleanup_local(self, allocation: LocalAllocation) -> None:
        """Unmap, release, and free a local HIP VMem allocation."""
        self._check_initialized()
        steps = [
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
        """Return the HIP VMem allocation granularity for this device."""
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
            raise LocalHipError("hipMemAddressReserve returned a null VA")
        return int(reserved.value)

    def free_va(self, va: int, size: int) -> None:
        """Free a HIP VA range previously returned by reserve_va."""
        self._check_initialized()
        _hip_try(_hip.hipMemAddressFree(ctypes.c_void_p(va), size), "hipMemAddressFree")

    def get_address_range(self, ptr: int) -> tuple[int, int]:
        """Return the base allocation range containing a HIP device pointer."""
        self._check_initialized()
        base_ptr = ctypes.c_void_p()
        base_size = ctypes.c_size_t()
        _hip_try(
            _hip.hipMemGetAddressRange(
                ctypes.byref(base_ptr),
                ctypes.byref(base_size),
                ctypes.c_void_p(int(ptr)),
            ),
            "hipMemGetAddressRange",
        )
        if base_ptr.value is None:
            raise LocalHipError("hipMemGetAddressRange returned a null base pointer")
        return int(base_ptr.value), int(base_size.value)
