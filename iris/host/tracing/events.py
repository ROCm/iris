# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Trace event type IDs and Triton-side enumeration.

EVENT_NAMES and TraceEvent must stay in sync: same IDs for the same operations.

Event ID ranges:
    0–1023      iris ops        (data movement, atomics)
    1024–2047   user data movement  (fetch/prefetch)
    2048–3071   user compute        (compute, reduce)
    3072–4095   synchronization     (wait, barrier)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


# Event type IDs to names mapping (used for export / display).
# Keep in sync with TraceEvent below.
EVENT_NAMES = {
    # iris ops (0–1023)
    0: "load",
    1: "store",
    2: "get",
    3: "put",
    4: "copy",
    5: "atomic_add",
    6: "atomic_sub",
    7: "atomic_cas",
    8: "atomic_xchg",
    9: "atomic_xor",
    10: "atomic_and",
    11: "atomic_or",
    12: "atomic_min",
    13: "atomic_max",
    # User data movement (1024–2047)
    1024: "fetch",
    # User compute (2048–3071)
    2048: "compute",
    2049: "reduce",
    # Synchronization (3072–4095)
    3072: "wait",
    3073: "barrier",
}


@aggregate
class TraceEvent:
    """
    Trace event type enumeration for iris operations and kernel instrumentation.

    Event ID ranges:
        0–1023      iris ops            (data movement, atomics)
        1024–2047   user data movement  (fetch/prefetch)
        2048–3071   user compute        (compute, reduce)
        3072–4095   synchronization     (wait, barrier)

    Usage:
        >>> ctx.record_event(event_id=TraceEvent().put, target_rank=1, address=ptr)

    Available event types:
        iris ops (0–1023):
        - load (0): Remote load operation
        - store (1): Remote store operation
        - get (2): Remote read (pull from remote to local)
        - put (3): Remote write (push from local to remote)
        - copy (4): Peer-to-peer copy between ranks
        - atomic_add (5) .. atomic_max (13): Atomic operations

        User data movement (1024–2047):
        - fetch (1024): Prefetching / staging data

        User compute (2048–3071):
        - compute (2048): Kernel compute phase (GEMM, FFT, etc.)
        - reduce (2049): Reduction operation

        Synchronization (3072–4095):
        - wait (3072): Stalled on a dependency
        - barrier (3073): Synchronization point
    """

    # iris ops (0–1023)
    load: tl.constexpr
    store: tl.constexpr
    get: tl.constexpr
    put: tl.constexpr
    copy: tl.constexpr
    atomic_add: tl.constexpr
    atomic_sub: tl.constexpr
    atomic_cas: tl.constexpr
    atomic_xchg: tl.constexpr
    atomic_xor: tl.constexpr
    atomic_and: tl.constexpr
    atomic_or: tl.constexpr
    atomic_min: tl.constexpr
    atomic_max: tl.constexpr

    # User data movement (1024–2047)
    fetch: tl.constexpr

    # User compute (2048–3071)
    compute: tl.constexpr
    reduce: tl.constexpr

    # Synchronization (3072–4095)
    wait: tl.constexpr
    barrier: tl.constexpr

    @triton.constexpr_function
    def __init__(self):
        # iris ops (0–1023)
        self.load = tl.constexpr(0)
        self.store = tl.constexpr(1)
        self.get = tl.constexpr(2)
        self.put = tl.constexpr(3)
        self.copy = tl.constexpr(4)
        self.atomic_add = tl.constexpr(5)
        self.atomic_sub = tl.constexpr(6)
        self.atomic_cas = tl.constexpr(7)
        self.atomic_xchg = tl.constexpr(8)
        self.atomic_xor = tl.constexpr(9)
        self.atomic_and = tl.constexpr(10)
        self.atomic_or = tl.constexpr(11)
        self.atomic_min = tl.constexpr(12)
        self.atomic_max = tl.constexpr(13)

        # User data movement (1024–2047)
        self.fetch = tl.constexpr(1024)

        # User compute (2048–3071)
        self.compute = tl.constexpr(2048)
        self.reduce = tl.constexpr(2049)

        # Synchronization (3072–4095)
        self.wait = tl.constexpr(3072)
        self.barrier = tl.constexpr(3073)
