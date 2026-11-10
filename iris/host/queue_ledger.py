# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
What triggered plans (iris.experimental.plan) have armed on this process's host copy-engine queues.

The queues belong to the process, not to an Iris instance, so the ledger does too. A queue is keyed
``(src, dst, queue)``: ``queue`` is the host queue index within the GPU pair (rocm-xio's channel,
one per lane, see ``iris.experimental.plan.queues``). Byte counts are keyed by peer (queue 0) or by
``(peer, queue)``.
"""

from collections import defaultdict

from iris.device.sdma_utils import bind_sdma_ep


def _key(src, dst):
    """``(src, peer, queue)`` from a byte-count key: a peer (queue 0) or ``(peer, queue)``."""
    peer, queue = dst if isinstance(dst, tuple) else (dst, 0)
    return src, peer, queue


def _queue_name(peer, queue):
    return f"queue to rank {peer}" + (f" (host queue {queue})" if queue else "")


class QueueLedger:
    """What is prepared on each of this process's host queues, in the order it was prepared."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = defaultdict(list)  # (src, dst, queue) -> [(plan, round, nbytes)]
        self.holds = defaultdict(dict)  # (src, dst, queue) -> {plan: nbytes}, space kept for a plan's rounds

    def _used(self, key):
        held = self.holds[key]
        return sum(held.values()) + sum(n for plan, _, n in self.entries[key] if plan not in held)

    def _check(self, src, nbytes):
        for dst, n in nbytes.items():
            key = _key(src, dst)
            used = self._used(key)
            if used + n > self.capacity:
                raise RuntimeError(
                    f"preparing {n} bytes of copies on the {_queue_name(*key[1:])} would exceed {self.capacity} bytes "
                    f"({used} already prepared); wait for or close other plans first"
                )

    def hold(self, plan, src, nbytes):
        """Keep ``nbytes[peer]`` of each queue for ``plan``, so its rounds can be prepared without checking again."""
        self._check(src, nbytes)
        for dst, n in nbytes.items():
            self.holds[_key(src, dst)][plan] = n

    def reserve(self, plan, round, src, nbytes):
        for dst, n in nbytes.items():
            key = _key(src, dst)
            held = self.holds[key].get(plan)
            if held is not None and sum(e[2] for e in self.entries[key] if e[0] is plan) + n > held:
                raise RuntimeError(f"preparing round {round} would exceed the queue space held for the plan")
        self._check(src, {d: n for d, n in nbytes.items() if plan not in self.holds[_key(src, d)]})
        for dst, n in nbytes.items():
            self.entries[_key(src, dst)].append((plan, round, n))

    def release(self, plan, round=None):
        """Forget ``plan``'s prepared ``round``, or all of its rounds and held space."""
        for key in self.entries:
            self.entries[key] = [
                e for e in self.entries[key] if e[0] is not plan or (round is not None and e[1] != round)
            ]
        if round is None:
            for held in self.holds.values():
                held.pop(plan, None)

    def earlier(self, plan):
        """(plan, round) prepared before ``plan``'s last round, on the queues it uses."""
        found = []
        for entries in self.entries.values():
            mine = [i for i, e in enumerate(entries) if e[0] is plan]
            if mine:
                found += [(e[0], e[1]) for e in entries[: mine[-1]] if e[0] is not plan]
        return found

    def plans(self):
        """Every plan with copies prepared, each once."""
        return list(dict.fromkeys(e[0] for entries in self.entries.values() for e in entries))

    def check_order(self, plan, round, started, action):
        """Raise unless every round prepared before ``plan``'s ``round`` has ``started(other, round)``."""
        for (_, peer, queue), entries in self.entries.items():
            if not any(e[0] is plan and e[1] == round for e in entries):
                continue  # nothing of the round is queued there, so nothing there can hold it back
            for other, other_round, _ in entries:
                if other is plan and other_round == round:
                    break
                if other is not plan and not started(other, other_round):
                    raise RuntimeError(
                        f"plan prepared earlier on the {_queue_name(peer, queue)} (round {other_round}) has not started it; "
                        f"plans sharing queues must {action} in the order they were prepared"
                    )

    def check_idle(self, src, dst, queue=0):
        if self.entries.get((src, dst, queue)):
            raise RuntimeError(
                f"triggered plans have copies prepared on the copy-engine {_queue_name(dst, queue)}; "
                "wait for or close those plans before using it"
            )


_ledger = None


def ledger():
    global _ledger
    if _ledger is None:
        # Prepared copies hold their ring space until their gate opens, and rocm-xio's host submit
        # spins until the ring has room: one that does not fit behind them would hang the host before
        # any kernel opens a gate. Plans get half the ring; the rest covers wrap padding, work queued
        # before them, and rounds forgotten at next_round() whose copies are still draining (their
        # producers are already queued). Half is a conservative margin, not a measured bound.
        _ledger = QueueLedger(bind_sdma_ep().SDMA_QUEUE_SIZE // 2)
    return _ledger
