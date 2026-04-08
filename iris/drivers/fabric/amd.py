# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
AMD fabric driver stub.
"""

from __future__ import annotations

from iris.drivers.base import BaseFabricDriver, DriverNotSupported, LocalAllocation, PeerMapping

__all__ = ["AmdFabricDriver"]

_NOT_IMPLEMENTED_MESSAGE = "AMD fabric driver not yet implemented"


class AmdFabricDriver(BaseFabricDriver):
    """AMD fabric driver placeholder."""

    def initialize(self, device_ordinal: int) -> None:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def allocate_exportable(self, size: int) -> LocalAllocation:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def export_handle(self, allocation: LocalAllocation) -> bytes:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def import_and_map(self, peer_rank: int, handle_bytes: bytes, size: int) -> PeerMapping:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def cleanup_import(self, mapping: PeerMapping) -> None:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)

    def cleanup_local(self, allocation: LocalAllocation) -> None:
        raise DriverNotSupported(_NOT_IMPLEMENTED_MESSAGE)
