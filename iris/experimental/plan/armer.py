# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Arms triggered-plan chains on the host-initiated copy-engine queue through rocm-xio.

Per (chunk, peer): POLL_REGMEM(counter >= writers * round) -> one copy per region -> ATOMIC ADD_RTN_32(arrival += 1).
After a round's last chain to a peer: ATOMIC ADD_RTN_32(completion += 1), on this rank.
"""

from dataclasses import dataclass

from iris.device.sdma_utils import bind_sdma_ep

CHANNEL = 0
POLL_REGMEM_BYTES = 24


@dataclass(frozen=True)
class Copy:
    src: int
    dst: int
    width: int
    height: int
    pitch: int
    dst_pitch: int


def chain_bytes(chunk):
    sdma_ep = bind_sdma_ep()
    copy_bytes = sdma_ep.COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES if chunk.is_2d else sdma_ep.COPY_LINEAR_COMMAND_BYTES
    return POLL_REGMEM_BYTES + len(chunk.regions) * copy_bytes + sdma_ep.ATOMIC_COMMAND_BYTES


def round_bytes(schedule, sender, world_size):
    """Bytes one round of ``sender``'s ``schedule`` takes on each peer's queue: its chains, then the completion signal."""
    signal_bytes = bind_sdma_ep().ATOMIC_COMMAND_BYTES
    nbytes = {}
    for chunk in schedule:
        chain = chain_bytes(chunk)
        for peer in chunk.recipients(sender, world_size):
            nbytes[peer] = nbytes.get(peer, signal_bytes) + chain
    return nbytes


class Chain:
    """One (chunk, peer) chain's arguments, fixed for the plan's life: each round re-arms it with a new threshold."""

    def __init__(self, src_device, dst_device, counter_addr, copies, arrival_addr, is_2d):
        self.src_device, self.dst_device = src_device, dst_device
        self.counter_addr, self.arrival_addr = counter_addr, arrival_addr
        self.copies = copies
        self.tiles = None
        if is_2d:
            # One-byte elements, so widths and pitches stay in bytes: destinations, sources, element size,
            # heights, widths, destination pitches, source pitches
            self.tiles = (
                [c.dst for c in copies],
                [c.src for c in copies],
                1,
                [c.height for c in copies],
                [c.width for c in copies],
                [c.dst_pitch for c in copies],
                [c.pitch for c in copies],
            )


def arm_chain(chain, threshold):
    """Arm ``chain`` on the host queue: its copies wait until its counter reaches ``threshold``."""
    sdma_ep = bind_sdma_ep()
    src, dst = chain.src_device, chain.dst_device
    if chain.tiles is not None:
        sdma_ep.wait_flag_then_put_tiles(src, dst, CHANNEL, chain.counter_addr, threshold, *chain.tiles)
    else:
        first, *rest = chain.copies
        sdma_ep.wait_flag_then_put(
            src, dst, CHANNEL, chain.counter_addr, threshold, first.dst, first.src, first.width, 32
        )
        for c in rest:
            sdma_ep.put(src, dst, CHANNEL, c.dst, c.src, c.width)
    sdma_ep.signal(src, dst, CHANNEL, chain.arrival_addr, 1, 32)


def signal_completion(src_device, dst_device, completion_addr):
    """Queue ``completion += 1`` behind every chain armed so far from ``src_device`` to ``dst_device``."""
    bind_sdma_ep().signal(src_device, dst_device, CHANNEL, completion_addr, 1, 32)
