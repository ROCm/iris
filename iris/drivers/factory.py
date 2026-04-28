# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Driver factory: vendor + interconnect -> BaseDriver."""

from __future__ import annotations

from iris.drivers.base import BaseDriver, DriverNotSupported
from iris.drivers.fabric.amd import AmdFabricDriver
from iris.drivers.fabric.nvidia import NvidiaFabricDriver
from iris.drivers.local.amd import LocalHipDriver
from iris.drivers.local.nvidia import LocalCudaDriver
from iris.host.distributed.topology import InterconnectLevel

__all__ = ["DriverFactory"]


class DriverFactory:
    """Stateless factory for memory drivers."""

    @staticmethod
    def create_driver(vendor: str, interconnect: InterconnectLevel) -> BaseDriver:
        v = vendor.strip().lower()
        if v == "nvidia":
            if interconnect == InterconnectLevel.INTRA_RACK_FABRIC:
                return NvidiaFabricDriver()
            if interconnect == InterconnectLevel.INTRA_NODE:
                return LocalCudaDriver()
        elif v == "amd":
            if interconnect == InterconnectLevel.INTRA_RACK_FABRIC:
                return AmdFabricDriver()
            if interconnect == InterconnectLevel.INTRA_NODE:
                return LocalHipDriver()
        raise DriverNotSupported(
            f"No driver for vendor={vendor!r}, interconnect={interconnect!r}"
        )
