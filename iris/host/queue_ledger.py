# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
What triggered plans (iris.experimental.plan) have armed on this process's host copy-engine queues.

The queues belong to the process, not to an Iris instance, so the ledger does too.
"""

from collections import defaultdict

from iris.device.sdma_utils import bind_sdma_ep


class QueueLedger:
    """What is armed on each of this process's host queues, in arming order."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = defaultdict(list)  # (src, dst) -> [(plan, round, nbytes)]

    def reserve(self, plan, round, src, nbytes):
        for peer, n in nbytes.items():
            used = sum(e[2] for e in self.entries[(src, peer)])
            if used + n > self.capacity:
                raise RuntimeError(
                    f"preparing {n} bytes of copies on the queue to rank {peer} would exceed {self.capacity} bytes "
                    f"({used} already prepared); wait for or close other plans first"
                )
        for peer, n in nbytes.items():
            self.entries[(src, peer)].append((plan, round, n))

    def release(self, plan):
        for key in self.entries:
            self.entries[key] = [e for e in self.entries[key] if e[0] is not plan]

    def earlier(self, plan):
        """Plans armed before ``plan`` on the queues it uses."""
        found = []
        for entries in self.entries.values():
            plans = [e[0] for e in entries]
            for i, other in enumerate(plans):
                if other is plan:
                    found += plans[:i]
                    break
        return found

    def plans(self):
        """Every plan with chains armed, each once."""
        return list({id(e[0]): e[0] for entries in self.entries.values() for e in entries}.values())

    def check_order(self, plan, allowed, action):
        for (_, peer), entries in self.entries.items():
            for other, round, _ in entries:
                if other is plan:
                    break
                if other.state not in allowed:
                    raise RuntimeError(
                        f"plan prepared earlier on the queue to rank {peer} (round {round}) is {other.state.value}; "
                        f"plans sharing queues must {action} in the order they were prepared"
                    )

    def check_idle(self, src, dst):
        if self.entries.get((src, dst)):
            raise RuntimeError(
                f"triggered plans have copies prepared on the copy-engine queue to rank {dst}; "
                "wait for or close those plans before using it"
            )


_ledger = None


def ledger():
    global _ledger
    if _ledger is None:
        # Prepared copies hold their ring space until their gate opens, and rocm-xio's host submit
        # spins until the ring has room: one that does not fit behind them would hang the host before
        # any kernel opens a gate. Plans get half the ring; the rest covers wrap padding and work
        # queued before them. Half is a conservative margin, not a measured bound.
        _ledger = QueueLedger(bind_sdma_ep().SDMA_QUEUE_SIZE // 2)
    return _ledger
