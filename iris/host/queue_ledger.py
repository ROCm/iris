# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
What iris.triggered plans have armed on this process's host copy-engine queues.

The queues belong to the process, not to an Iris instance, so the ledger does too.
"""

from collections import defaultdict

from xio import sdma_ep


class QueueLedger:
    """What is armed on each of this process's host queues, in arming order."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = defaultdict(list)  # (src, dst) -> [(plan, epoch, nbytes)]

    def reserve(self, plan, epoch, src, nbytes):
        for peer, n in nbytes.items():
            used = sum(e[2] for e in self.entries[(src, peer)])
            if used + n > self.capacity:
                raise RuntimeError(
                    f"arming {n} bytes on the queue to rank {peer} would exceed {self.capacity} armed bytes "
                    f"({used} already armed); complete or destroy other plans first"
                )
        for peer, n in nbytes.items():
            self.entries[(src, peer)].append((plan, epoch, n))

    def release(self, plan):
        for key in self.entries:
            self.entries[key] = [e for e in self.entries[key] if e[0] is not plan]

    def check_order(self, plan, allowed, action):
        for (_, peer), entries in self.entries.items():
            for other, epoch, _ in entries:
                if other is plan:
                    break
                if other.state not in allowed:
                    raise RuntimeError(
                        f"plan armed earlier on the queue to rank {peer} (epoch {epoch}) is {other.state.value}; "
                        f"plans sharing queues must {action} in the order they were armed"
                    )

    def check_idle(self, src, dst):
        if self.entries.get((src, dst)):
            raise RuntimeError(
                f"triggered chains are armed on the copy-engine queue to rank {dst}; "
                "complete, abort or destroy those plans before using it"
            )


_ledger = None


def ledger():
    global _ledger
    if _ledger is None:
        # Leave half the ring for wrap padding and work queued before the plan
        _ledger = QueueLedger(sdma_ep.SDMA_QUEUE_SIZE // 2)
    return _ledger
