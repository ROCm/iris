# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# The queue ledger on its own: no GPU or copy engine involved.

import pytest

from iris.host.queue_ledger import QueueLedger


class _Plan:
    """A stand-in plan: hashable by identity, like TriggeredPlan."""

    def __init__(self, round, state):
        self.round, self.state = round, state


def _plan(round=1, state="running"):
    return _Plan(round, state)


def _started(plan, round):
    return plan.state in ("running", "done") and round == plan.round


def test_hold_counts_against_capacity():
    ledger = QueueLedger(100)
    a, b = _plan(), _plan()
    ledger.hold(a, 0, {(1, 0): 60})
    with pytest.raises(RuntimeError, match="would exceed"):
        ledger.hold(b, 0, {(1, 0): 60})
    with pytest.raises(RuntimeError, match="would exceed"):
        ledger.reserve(b, 1, 0, {(1, 0): 41})
    ledger.reserve(b, 1, 0, {(1, 0): 40})


def test_rounds_within_a_hold_need_no_capacity_check():
    ledger = QueueLedger(100)
    a, b = _plan(), _plan()
    ledger.hold(a, 0, {(1, 0): 60})
    ledger.reserve(b, 1, 0, {(1, 0): 40})  # the queue is now full
    ledger.reserve(a, 1, 0, {(1, 0): 30})
    ledger.reserve(a, 2, 0, {(1, 0): 30})
    with pytest.raises(RuntimeError, match="held for the plan"):
        ledger.reserve(a, 3, 0, {(1, 0): 30})
    ledger.release(a, 1)
    ledger.reserve(a, 3, 0, {(1, 0): 30})
    assert [(e[1], e[2]) for e in ledger.entries[(0, 1, 0)] if e[0] is a] == [(2, 30), (3, 30)]


def test_release_whole_plan_drops_its_hold():
    ledger = QueueLedger(100)
    a = _plan()
    ledger.hold(a, 0, {(1, 0): 60, (2, 0): 60})
    ledger.reserve(a, 1, 0, {(1, 0): 30, (2, 0): 30})
    ledger.release(a)
    assert not any(ledger.entries.values())
    assert not any(ledger.holds.values())
    ledger.hold(_plan(), 0, {(1, 0): 100, (2, 0): 100})


def test_earlier_and_check_order_are_per_round():
    ledger = QueueLedger(100)
    a, b = _plan(round=1, state="running"), _plan(round=1, state="prepared")
    ledger.reserve(a, 1, 0, {(1, 0): 10})
    ledger.reserve(b, 1, 0, {(1, 0): 10})
    ledger.reserve(a, 2, 0, {(1, 0): 10})  # a prepared its next round behind b's first
    assert ledger.earlier(b) == [(a, 1)]
    assert ledger.earlier(a) == [(b, 1)]
    ledger.check_order(b, 1, _started, "start")  # a's round 1 has started

    a.state = "done"
    ledger.release(a, 1)
    a.round, a.state = 2, "prepared"
    with pytest.raises(RuntimeError, match="prepared earlier"):
        ledger.check_order(a, 2, _started, "start")  # behind b's round 1, which has not started
    b.state = "running"
    ledger.check_order(a, 2, _started, "start")


def test_budget_is_per_lane():
    ledger = QueueLedger(100)
    ledger.reserve(_plan(), 1, 0, {(1, 0): 90})
    ledger.reserve(_plan(), 1, 0, {(1, 1): 90, (2, 0): 90})
    with pytest.raises(RuntimeError, match=r"on the queue 1 to rank 1 would exceed"):
        ledger.reserve(_plan(), 1, 0, {(1, 1): 20})


def test_lanes_do_not_order_each_other():
    ledger = QueueLedger(1000)
    first, second = _plan(state="prepared"), _plan(state="prepared")
    ledger.reserve(first, 1, 0, {(1, 0): 10})
    ledger.reserve(second, 1, 0, {(1, 1): 10})
    ledger.check_order(second, 1, _started, "start")


def test_idle_is_per_lane():
    ledger = QueueLedger(1000)
    plan = _plan()
    ledger.reserve(plan, 1, 0, {(1, 1): 10})
    ledger.check_idle(0, 1, 0)
    with pytest.raises(RuntimeError, match="copies prepared on the copy-engine queue 1 to rank 1"):
        ledger.check_idle(0, 1, 1)
    ledger.release(plan)
    ledger.check_idle(0, 1, 1)
