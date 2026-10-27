# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Arms triggered-plan chains on the host-initiated copy-engine queue through rocm-xio.

Per (chunk, peer): POLL_REGMEM(ready >= round) -> one copy per region -> ATOMIC ADD_RTN_64(arrival += 1).
"""

import ctypes
from dataclasses import dataclass

from iris.device.sdma_utils import bind_sdma_ep

CHANNEL = 0
POLL_REGMEM_BYTES = 24

# nanobind passes void* as a PyCapsule named "nb_handle"
_NB_HANDLE = b"nb_handle"
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _void_ptr(addr):
    return _capsule_new(addr, _NB_HANDLE, None)


@dataclass(frozen=True)
class Copy:
    src: int
    dst: int
    width: int
    height: int
    pitch: int


def chain_bytes(chunk):
    sdma_ep = bind_sdma_ep()
    copy_bytes = sdma_ep.COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES if chunk.is_2d else sdma_ep.COPY_LINEAR_COMMAND_BYTES
    return POLL_REGMEM_BYTES + len(chunk.regions) * copy_bytes + sdma_ep.ATOMIC_COMMAND_BYTES


def arm_chain(src_device, dst_device, ready_addr, round, copies, arrival_addr, is_2d):
    """Arm one chain on the host queue from ``src_device`` to ``dst_device``."""
    sdma_ep = bind_sdma_ep()
    if is_2d:
        tiles = []
        for c in copies:
            tile = sdma_ep.Tile()
            tile.data = _void_ptr(c.src)
            tile.pid_m = tile.pid_n = 0
            tile.block_m, tile.block_n = c.height, c.width
            tile.elem_size = 1
            tile.src_stride = c.pitch
            tiles.append(tile)
        sdma_ep.wait_flag_then_put_tiles(
            src_device,
            dst_device,
            CHANNEL,
            ready_addr,
            round,
            tiles,
            [c.dst for c in copies],
            [c.pitch for c in copies],
        )
    else:
        first, *rest = copies
        sdma_ep.wait_flag_then_put(
            src_device, dst_device, CHANNEL, ready_addr, round, first.src, first.dst, first.width, 32
        )
        for c in rest:
            sdma_ep.put(src_device, dst_device, CHANNEL, c.src, c.dst, c.width)
    sdma_ep.signal(src_device, dst_device, CHANNEL, arrival_addr, 1, 64)
