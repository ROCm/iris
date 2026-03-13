"""
Trace event type IDs and Triton-side enumeration.

EVENT_NAMES and TraceEvent must stay in sync: same IDs for the same operations.
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate


# Event type IDs to names mapping (used for export / display).
# Keep in sync with TraceEvent below.
EVENT_NAMES = {
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
}


@aggregate
class TraceEvent:
    """
    Trace event type enumeration for iris remote memory operations.

    Usage:
        >>> ctx.record_event(event_id=TraceEvent().put, target_rank=1, address=ptr)

    Available event types:
        Data Movement:
        - load (0): Remote load operation
        - store (1): Remote store operation
        - get (2): Remote read (pull from remote to local)
        - put (3): Remote write (push from local to remote)
        - copy (4): Peer-to-peer copy between ranks

        Atomic Operations:
        - atomic_add (5): Atomic addition
        - atomic_sub (6): Atomic subtraction
        - atomic_cas (7): Atomic compare-and-swap
        - atomic_xchg (8): Atomic exchange
        - atomic_xor (9): Atomic XOR
        - atomic_and (10): Atomic AND
        - atomic_or (11): Atomic OR
        - atomic_min (12): Atomic minimum
        - atomic_max (13): Atomic maximum
    """

    # Data movement operations
    load: tl.constexpr
    store: tl.constexpr
    get: tl.constexpr
    put: tl.constexpr
    copy: tl.constexpr

    # Atomic operations
    atomic_add: tl.constexpr
    atomic_sub: tl.constexpr
    atomic_cas: tl.constexpr
    atomic_xchg: tl.constexpr
    atomic_xor: tl.constexpr
    atomic_and: tl.constexpr
    atomic_or: tl.constexpr
    atomic_min: tl.constexpr
    atomic_max: tl.constexpr

    @triton.constexpr_function
    def __init__(self):
        # Data movement
        self.load = tl.constexpr(0)
        self.store = tl.constexpr(1)
        self.get = tl.constexpr(2)
        self.put = tl.constexpr(3)
        self.copy = tl.constexpr(4)

        # Atomics
        self.atomic_add = tl.constexpr(5)
        self.atomic_sub = tl.constexpr(6)
        self.atomic_cas = tl.constexpr(7)
        self.atomic_xchg = tl.constexpr(8)
        self.atomic_xor = tl.constexpr(9)
        self.atomic_and = tl.constexpr(10)
        self.atomic_or = tl.constexpr(11)
        self.atomic_min = tl.constexpr(12)
        self.atomic_max = tl.constexpr(13)
