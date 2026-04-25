# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base classes, shared dataclasses, and exceptions for fabric drivers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from iris.topology import InterconnectLevel

__all__ = [
    "PeerMapping",
    "LocalAllocation",
    "BaseFabricDriver",
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


class DriverError(RuntimeError):
    """Base exception for driver operations."""


class DriverNotSupported(DriverError):
    """The current hardware or software stack does not support this driver."""


class BaseFabricDriver(ABC):
    """Cross-node fabric memory sharing (for example NVSwitch or xGMI)."""

    @abstractmethod
    def initialize(self, device_ordinal: int) -> None:
        """Prepare the driver for a specific local GPU."""

    @abstractmethod
    def allocate_exportable(self, size: int) -> LocalAllocation:
        """Allocate memory that can be shared through the fabric transport."""

    @abstractmethod
    def export_handle(self, allocation: LocalAllocation) -> bytes:
        """Export a transport-specific handle for a local allocation."""

    @abstractmethod
    def import_and_map(self, peer_rank: int, handle_bytes: bytes, size: int) -> PeerMapping:
        """Import a peer handle and map it into the local virtual address space."""

    @abstractmethod
    def cleanup_import(self, mapping: PeerMapping) -> None:
        """Release a mapped peer allocation."""

    @abstractmethod
    def cleanup_local(self, allocation: LocalAllocation) -> None:
        """Release a locally-exported allocation."""
