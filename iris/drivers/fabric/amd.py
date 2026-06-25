# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
AMD fabric driver stub.
"""

from __future__ import annotations

from typing import Optional

from iris.drivers.base import (
    BaseDriver,
    DriverNotSupported,
    ExportableMemory,
    LocalAllocation,
    MappingPlacement,
    PeerMapping,
)

__all__ = ["AmdFabricDriver"]

_NOT_IMPLEMENTED_MESSAGE = "AMD fabric driver not yet implemented"


class AmdFabricDriver(BaseDriver):
    """AMD fabric driver placeholder."""

    def initialize(self, device_ordinal: int) -> None:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def allocate_exportable(
        self,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> LocalAllocation:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def export_handle(self, memory: ExportableMemory) -> bytes:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def import_and_map(
        self,
        peer_rank: int,
        handle_bytes: bytes,
        size: int,
        placement: Optional[MappingPlacement] = None,
    ) -> PeerMapping:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def cleanup(self, target: LocalAllocation | PeerMapping) -> None:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def get_minimum_granularity(self) -> int:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def reserve_va(self, size: int, alignment: int = 0) -> int:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def free_va(self, va: int, size: int) -> None:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)
