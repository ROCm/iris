# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Lanes: Chunk.lane, per-queue bytes and ledger, the device view's queue counts, and sdma_lanes on the
# context (host only; stand-ins for xio.sdma_ep and the context, no GPU).

from types import SimpleNamespace

import pytest
import torch

from iris import Iris
from iris.experimental.plan import Chunk, Region, armer, chunks, tile_chunks, whole
from iris.experimental.plan import triton as dv
from iris.experimental.plan.queues import queue_index
from iris.experimental.plan.schedule import validate
from iris.experimental.plan.triggered import view_words
from iris.host import queue_ledger
from iris.host.iris import _require_multi_channel, _validate_sdma_lanes
from iris.host.queue_ledger import QueueLedger

REGION = Region(0, 64)
SRC, DST = 0x1000, 0x9000


class FakeSdmaEp:
    COPY_LINEAR_COMMAND_BYTES = 32
    COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES = 48
    ATOMIC_COMMAND_BYTES = 36

    def __init__(self, channels=1, create_error=None):
        self.calls = []
        self.channels = channels
        self.create_error = create_error

    def create_host_queue(self, *args, **kwargs):
        self.calls.append(("create_host_queue", args, kwargs))
        if self.create_error:
            raise self.create_error

    def get_host_channel_count(self, *args):
        self.calls.append(("get_host_channel_count", args))
        return self.channels

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))

        return record


class FakeTensor:
    def __init__(self, ptr, nbytes):
        self._ptr, self._nbytes = ptr, nbytes

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1


class _Plan:
    def __init__(self, round=1, state="prepared"):
        self.round, self.state = round, state


def _started(plan, round):
    return plan.state in ("running", "done") and round == plan.round


def test_chunk_lane_defaults_to_zero_and_builders_pass_it():
    assert Chunk((REGION,)).lane == 0
    view = torch.zeros(64, 64, dtype=torch.int32)
    assert {c.lane for c in whole(view[0], base=view, lane=1)} == {1}
    assert {c.lane for c in chunks(view, 4, lane=2)} == {2}
    assert {c.lane for c in tile_chunks(view, (16, 16), 4, lane=3)} == {3}


def test_chunk_lane_must_be_an_integer():
    for lane in (True, 1.0, "1", None):
        with pytest.raises(ValueError, match="lane"):
            Chunk((REGION,), lane=lane)


def test_validate_accepts_only_the_context_lanes():
    two = [Chunk((REGION,), lane=0), Chunk((REGION,), lane=1)]
    assert validate(two, 64, lanes=2) == two
    with pytest.raises(ValueError, match="lane 1 is not one of the 1 lane"):
        validate(two, 64)
    with pytest.raises(ValueError, match="lane -1"):
        validate([Chunk((REGION,), lane=-1)], 64, lanes=2)


def test_lane_runs_on_the_queue_of_the_same_index():
    assert [queue_index(lane) for lane in range(4)] == [0, 1, 2, 3]


def test_round_bytes_are_per_peer_and_queue(monkeypatch):
    monkeypatch.setattr(armer, "bind_sdma_ep", FakeSdmaEp)
    schedule = [
        Chunk((Region(0, 64), Region(64, 64)), lane=0),
        Chunk((Region(128, 64, rows=4, stride_bytes=128),), lane=1, peers=(1,)),
        Chunk((Region(1024, 64),), lane=0),
    ]
    first, second, third = 24 + 2 * 32 + 36, 24 + 48 + 36, 24 + 32 + 36
    signal = 36
    assert armer.round_bytes(schedule, 0, 3) == {
        (1, 0): signal + first + third,
        (2, 0): signal + first + third,
        (1, 1): signal + second,
    }


def test_view_counts_the_queues_to_each_peer():
    schedules = [
        [Chunk((Region(0, 64),)), Chunk((Region(64, 64),), lane=1, peers=(1,))],
        [Chunk((Region(128, 64),))],
        [],
    ]
    words = view_words(0, 3, 0, schedules)
    per_peer = dv.HEADER_WORDS.value + 3 * dv.RECORD_WORDS.value
    assert words[dv.H_VERSION.value] == dv.ABI_VERSION
    assert words[per_peer : per_peer + 3] == [0, 2, 1]
    words = view_words(2, 3, 0, schedules)
    assert words[per_peer : per_peer + 3] == [0, 0, 0]


def test_ledger_budget_is_per_queue():
    ledger = QueueLedger(100)
    ledger.reserve(_Plan(), 1, 0, {1: 90})
    ledger.reserve(_Plan(), 1, 0, {(1, 1): 90, 2: 90})  # another queue, another peer: own budget
    with pytest.raises(RuntimeError, match=r"20 bytes of copies on the queue to rank 1 would exceed 100"):
        ledger.reserve(_Plan(), 1, 0, {1: 20})
    with pytest.raises(RuntimeError, match=r"on the queue to rank 1 \(host queue 1\) would exceed"):
        ledger.reserve(_Plan(), 1, 0, {(1, 1): 20})
    assert set(ledger.entries) == {(0, 1, 0), (0, 1, 1), (0, 2, 0)}


def test_plans_on_different_queues_do_not_wait_for_each_other():
    ledger = QueueLedger(1000)
    first, second = _Plan(), _Plan()
    ledger.reserve(first, 1, 0, {(1, 0): 10})
    ledger.reserve(second, 1, 0, {(1, 1): 10})
    ledger.check_order(second, 1, _started, "start")  # first sits on queue 0 only: not ahead of second


def test_plans_on_one_queue_keep_their_order():
    ledger = QueueLedger(1000)
    first, second = _Plan(), _Plan()
    ledger.reserve(first, 1, 0, {(1, 1): 10})
    ledger.reserve(second, 1, 0, {(1, 1): 10})
    with pytest.raises(RuntimeError, match=r"prepared earlier on the queue to rank 1 \(host queue 1\)"):
        ledger.check_order(second, 1, _started, "start")
    first.state = "running"
    ledger.check_order(second, 1, _started, "start")


def test_host_work_is_refused_only_on_a_queue_with_copies():
    ledger = QueueLedger(1000)
    plan = _Plan()
    ledger.reserve(plan, 1, 0, {(1, 1): 10})
    ledger.check_idle(0, 1)  # queue 0
    ledger.check_idle(0, 2, 1)  # another peer
    with pytest.raises(RuntimeError, match=r"prepared on the copy-engine queue to rank 1 \(host queue 1\)"):
        ledger.check_idle(0, 1, 1)
    ledger.release(plan)
    ledger.check_idle(0, 1, 1)


def test_queue_zero_messages_do_not_change():
    ledger = QueueLedger(10)
    ledger.reserve(_Plan(), 1, 0, {1: 10})
    with pytest.raises(RuntimeError) as busy:
        ledger.check_idle(0, 1)
    assert str(busy.value) == (
        "triggered plans have copies prepared on the copy-engine queue to rank 1; "
        "wait for or close those plans before using it"
    )
    with pytest.raises(RuntimeError) as full:
        ledger.reserve(_Plan(), 1, 0, {1: 1})
    assert str(full.value) == (
        "preparing 1 bytes of copies on the queue to rank 1 would exceed 10 bytes (10 already prepared); "
        "wait for or close other plans first"
    )


def _ctx(lanes=1, **ep_kwargs):
    ep = FakeSdmaEp(**ep_kwargs)
    idle = []
    ctx = SimpleNamespace(
        sdma_lanes=lanes,
        _sdma_ep=ep,
        gpu_id=0,
        _copy_engine_gpus=[0, 1],
        get_rank=lambda: 0,
        get_num_ranks=lambda: 2,
        _require_copy_engine=lambda: ep,
        _copy_engine_devices=lambda rank: (0, rank),
        _check_host_queue_idle=lambda rank, channel=0: idle.append((rank, channel)),
        heap=SimpleNamespace(translate=lambda ptr, src_rank, dst_rank: ptr + 0x4000_0000),
    )
    ctx._flag_pointer_and_bits = lambda flag, **kw: Iris._flag_pointer_and_bits(ctx, flag, **kw)
    return ctx, ep, idle


def test_sdma_lanes_must_be_a_positive_integer():
    for lanes in (0, -1, True, 2.0, "2", None):
        with pytest.raises(ValueError, match="positive integer"):
            _validate_sdma_lanes(lanes, ("sdma",))


def test_sdma_lanes_above_one_need_the_sdma_engine():
    _validate_sdma_lanes(1, ())
    _validate_sdma_lanes(3, ("sdma",))
    with pytest.raises(ValueError, match=r'engines=\("sdma",\)'):
        _validate_sdma_lanes(2, ())


def test_several_lanes_need_a_rocm_xio_that_can_create_channels():
    _require_multi_channel(FakeSdmaEp())
    with pytest.raises(RuntimeError, match="update rocm-xio"):
        _require_multi_channel(SimpleNamespace(create_host_queue=None))


def test_one_lane_creates_the_single_channel_every_rocm_xio_can():
    ctx, ep, _ = _ctx(lanes=1)
    Iris._create_host_queues(ctx, 0, 1)
    assert ep.calls == [("create_host_queue", (0, 1), {})]


def test_several_lanes_create_one_channel_each():
    ctx, ep, _ = _ctx(lanes=3, channels=3)
    Iris._create_host_queues(ctx, 0, 1)
    assert ep.calls == [("create_host_queue", (0, 1), {"num_channels": 3}), ("get_host_channel_count", (0, 1))]


def test_missing_channels_are_an_error():
    ctx, _, _ = _ctx(lanes=3, channels=2)
    with pytest.raises(RuntimeError, match="has 2 host channels, sdma_lanes=3"):
        Iris._create_host_queues(ctx, 0, 1)


def test_a_failed_channel_creation_names_the_request():
    ctx, _, _ = _ctx(lanes=4, create_error=RuntimeError("initHostQueue failed"))
    with pytest.raises(RuntimeError, match="could not create sdma_lanes=4 host channels from device 0 to device 1"):
        Iris._create_host_queues(ctx, 0, 1)


def test_idle_check_passes_the_queue_to_the_ledger(monkeypatch):
    ledger = QueueLedger(1000)
    monkeypatch.setattr(queue_ledger, "_ledger", ledger)
    ledger.reserve(_Plan(), 1, 0, {(1, 1): 10})
    ctx = SimpleNamespace(get_rank=lambda: 0)
    Iris._check_host_queue_idle(ctx, 1)
    with pytest.raises(RuntimeError, match=r"host queue 1"):
        Iris._check_host_queue_idle(ctx, 1, 1)


def test_host_puts_check_the_channel_they_use():
    ctx, ep, idle = _ctx(lanes=3)
    Iris.put(ctx, FakeTensor(SRC, 256), to_rank=1, channel=2, async_op=True)
    Iris.put_tile(ctx, SRC, 1, DST, 4, 8, 16, 64, 128, channel=1, async_op=True)
    Iris.put_tiles(ctx, [SRC], 1, [DST], 4, [8], [16], [64], [128], channel=2, async_op=True)
    assert idle == [(1, 2), (1, 1), (1, 2)]
    assert [call[1][2] for call in ep.calls] == [2, 1, 2]


def test_quiet_checks_the_channel_it_waits_on():
    ctx, ep, idle = _ctx(lanes=3)
    Iris.quiet(ctx, to_rank=1, channel=2)
    Iris.quiet(ctx, channel=1)
    assert idle == [(1, 2), (0, 1), (1, 1)]
    assert [c for c in ep.calls if c[0] == "quiet"] == [
        ("quiet", (0, 1, 2)),
        ("quiet", (0, 0, 1)),
        ("quiet", (0, 1, 1)),
    ]
