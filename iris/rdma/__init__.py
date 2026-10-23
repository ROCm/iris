# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""GPU-initiated RDMA for Iris: ``iris.iris(heap_size, engines=("rdma",), rdma_backend=...)``.

Iris's load/store model reaches the peers it can map. An RDMA backend reaches
the rest from inside a kernel, through a NIC. This package is where backends
plug in, so a kernel written against :mod:`iris.rdma.device` does not depend
on which one is underneath.

A backend is a class with this host interface (see :class:`RdmaBackend`):

- ``provides_heap``: whether the backend allocates the symmetric heap itself.
  RCCL GIN does: it can only send from and to memory registered as an RCCL
  window, and RCCL maps that window on same-node peers itself. A backend that
  instead registers memory Iris allocated (a NIC verbs backend, for example)
  sets this False and implements ``register_heap``.
- ``direct``: per peer, whether it is load/store reachable. Iris's ``store``,
  ``load``, atomics and collectives only reach those; every other peer must
  be reached with :mod:`iris.rdma.device`. Its ``heap_bases`` entry is 0.
- ``BACKEND_ID``: the constexpr the device functions dispatch on.
- ``extern_libs()``: the ``extern_libs`` launch kwarg for kernels that call
  :mod:`iris.rdma.device` (empty if the backend links nothing).
- Completion state is backend-specific; RCCL GIN adds ``device_comm``,
  ``device_context``, ``read_signals``, ``destroy_device_comm``.

Backends are registered by name in ``BACKENDS``, as ``"module:Class"`` so a
backend's dependencies load only when it is used.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

__all__ = ["BACKENDS", "HeapRegion", "RdmaBackend", "create_backend"]

# name -> "module:Class". The id each backend's device code dispatches on is in
# iris.rdma.device.
BACKENDS = {
    "rccl_gin": "iris.experimental.rccl_gin:RcclGinBackend",
}


@dataclass(frozen=True)
class HeapRegion:
    """Memory a backend provides as Iris's symmetric heap.

    ``tensor`` is the whole region (uint8). ``peer_bases[r]`` is where rank
    ``r``'s region is mapped in this process, for every ``direct`` peer, and
    this rank's own base for itself; 0 for the rest. The region cannot grow.
    """

    tensor: torch.Tensor
    peer_bases: list[int]
    direct: tuple[bool, ...]


class RdmaBackend:
    """Host interface of an RDMA backend. Collective methods must be called
    by every rank of the group the backend was created on."""

    name: str = ""
    BACKEND_ID: int = 0
    provides_heap: bool = False

    def __init__(self, group=None):
        raise NotImplementedError

    @property
    def direct(self) -> tuple[bool, ...]:
        raise NotImplementedError

    def allocate_heap(self, nbytes: int) -> HeapRegion:
        """Collective; only for ``provides_heap`` backends."""
        raise NotImplementedError(f"{self.name} does not provide the heap")

    def register_heap(self, base: int, nbytes: int) -> None:
        """Collective; only for backends that do not provide the heap."""
        raise NotImplementedError(f"{self.name} provides the heap; it does not register one")

    def extern_libs(self) -> dict:
        return {}

    def close(self) -> None:
        """Collective; release everything. Call before destroying the process group."""
        raise NotImplementedError


def create_backend(name: str, group=None) -> RdmaBackend:
    """Collective: construct the backend registered as ``name``."""
    try:
        target = BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown RDMA backend {name!r}. Available: {sorted(BACKENDS)}") from None
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(group=group)
