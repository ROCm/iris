# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base classes, shared dataclasses, and exceptions for memory drivers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from iris.host.distributed.topology import InterconnectLevel

__all__ = [
    "PeerMapping",
    "LocalAllocation",
    "CleanupTarget",
    "ExportableMemory",
    "MappingPlacement",
    "BaseDriver",
    "DriverError",
    "DriverNotSupported",
]


@dataclass
class PeerMapping:
    """A remote rank's memory mapped into this rank's address space."""

    peer_rank: int
    transport: InterconnectLevel
    remote_va: int
    size: int
    _driver_handle: Any = None


@dataclass
class LocalAllocation:
    """This rank's exportable allocation."""

    va: int
    size: int
    handle: Any
    _va_owned: bool = True


CleanupTarget = LocalAllocation | PeerMapping


@dataclass(frozen=True)
class ExportableMemory:
    """A local memory range that can be exported to peers."""

    va: int
    size: int
    allocation: Optional[LocalAllocation] = None


@dataclass(frozen=True)
class MappingPlacement:
    """Caller-reserved virtual address placement for a VMM mapping.

    If arena_base is set, the mapping belongs to a larger reserved VA arena.
    Drivers that need cumulative access programming can use that arena context
    without exposing access-range details in the public API.
    """

    va: int
    arena_base: Optional[int] = None

    def access_range(self, mapped_size: int, cumulative: bool = False) -> tuple[int, int]:
        if not cumulative or self.arena_base is None:
            return int(self.va), int(mapped_size)

        offset = int(self.va) - int(self.arena_base)
        if offset < 0:
            raise ValueError(f"placement va {self.va} is below arena base {self.arena_base}")
        return int(self.arena_base), offset + int(mapped_size)


class DriverError(RuntimeError):
    """Base exception for driver operations."""


class DriverNotSupported(DriverError):
    """The current hardware or software stack does not support this driver."""


class BaseDriver(ABC):
    """Generic base class for local and fabric memory drivers."""

    @abstractmethod
    def initialize(self, device_ordinal: int) -> None:
        """Prepare the driver for a specific local GPU."""

    @abstractmethod
    def allocate_exportable(
        self,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> LocalAllocation:
        """Allocate exportable memory, optionally mapping it at a caller-reserved VA."""

    @abstractmethod
    def export_handle(self, memory: ExportableMemory) -> bytes:
        """Export a transport-specific handle for a local memory range."""

    @abstractmethod
    def import_and_map(
        self,
        peer_rank: int,
        handle_bytes: bytes,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> PeerMapping:
        """Import a peer handle and map it into the local virtual address space."""

    @abstractmethod
    def cleanup(self, target: CleanupTarget) -> None:
        """Release a local allocation or imported peer mapping."""

    @abstractmethod
    def get_minimum_granularity(self) -> int:
        """Minimum allocation granularity in bytes for this driver+device."""

    @abstractmethod
    def reserve_va(self, size: int, alignment: int = 0) -> int:
        """Reserve a virtual address range without backing physical memory."""

    @abstractmethod
    def free_va(self, va: int, size: int) -> None:
        """Free a VA range previously returned by reserve_va."""

    def get_address_range(self, ptr: int) -> tuple[int, int]:
        """Return the base VA and size of the allocation containing ptr."""
        raise DriverNotSupported(f"{type(self).__name__} does not support get_address_range")
