# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# The queue ledger on its own: no GPU or copy engine involved.

from types import SimpleNamespace

import pytest

from iris.host.queue_ledger import QueueLedger


def _plan(round=1, state="running"):
    return SimpleNamespace(round=round, state=state)


def _started(plan, round):
    return plan.state in ("running", "done") and round == plan.round


def test_hold_counts_against_capacity():
    ledger = QueueLedger(100)
    a, b = _plan(), _plan()
    ledger.hold(a, 0, {1: 60})
    with pytest.raises(RuntimeError, match="would exceed"):
        ledger.hold(b, 0, {1: 60})
    with pytest.raises(RuntimeError, match="would exceed"):
        ledger.reserve(b, 1, 0, {1: 41})
    ledger.reserve(b, 1, 0, {1: 40})


def test_rounds_within_a_hold_need_no_capacity_check():
    ledger = QueueLedger(100)
    a, b = _plan(), _plan()
    ledger.hold(a, 0, {1: 60})
    ledger.reserve(b, 1, 0, {1: 40})  # the queue is now full
    ledger.reserve(a, 1, 0, {1: 30})
    ledger.reserve(a, 2, 0, {1: 30})
    with pytest.raises(RuntimeError, match="held for the plan"):
        ledger.reserve(a, 3, 0, {1: 30})
    ledger.release(a, 1)
    ledger.reserve(a, 3, 0, {1: 30})
    assert [(e[1], e[2]) for e in ledger.entries[(0, 1)] if e[0] is a] == [(2, 30), (3, 30)]


def test_release_whole_plan_drops_its_hold():
    ledger = QueueLedger(100)
    a = _plan()
    ledger.hold(a, 0, {1: 60, 2: 60})
    ledger.reserve(a, 1, 0, {1: 30, 2: 30})
    ledger.release(a)
    assert not any(ledger.entries.values())
    assert not any(ledger.holds.values())
    ledger.hold(_plan(), 0, {1: 100, 2: 100})


def test_earlier_and_check_order_are_per_round():
    ledger = QueueLedger(100)
    a, b = _plan(round=1, state="running"), _plan(round=1, state="prepared")
    ledger.reserve(a, 1, 0, {1: 10})
    ledger.reserve(b, 1, 0, {1: 10})
    ledger.reserve(a, 2, 0, {1: 10})  # a prepared its next round behind b's first
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
