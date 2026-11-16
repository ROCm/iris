# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Copy-engine signal batches built from tritonBLAS's published tile schedule (host only).

from types import SimpleNamespace

import pytest
import tritonblas

from iris.ops.tritonblas_tile_schedule import (
    TileRect,
    build_launch_wave_signal_plan,
    coalesce_tiles,
    plan_gates,
    plan_row_batches,
    selector_schedule,
)


def _rect_tiles(rect):
    return {
        (m, n)
        for m in range(rect.m_tile_start, rect.m_tile_start + rect.m_tile_count)
        for n in range(rect.n_tile_start, rect.n_tile_start + rect.n_tile_count)
    }


def _selector(block_m=256, block_n=256, group_m=4, num_xcds=8):
    # schedule_for_selector reads only these attributes.
    return SimpleNamespace(block_m=block_m, block_n=block_n, block_k=64, group_m=group_m, num_sms=num_xcds)


def test_coalesce_full_block():
    tiles = [(m, n) for m in range(2, 6) for n in range(3, 7)]
    assert coalesce_tiles(tiles) == (TileRect(2, 3, 4, 4),)


def test_coalesce_ragged():
    # Column 0 has two runs; columns 1-2 share the run [0, 2); column 4 is not adjacent to column 2.
    tiles = [(0, 0), (1, 0), (3, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 4), (1, 4)]
    rects = coalesce_tiles(tiles)
    assert sorted(rects, key=lambda r: (r.n_tile_start, r.m_tile_start)) == [
        TileRect(0, 0, 2, 3),
        TileRect(3, 0, 1, 1),
        TileRect(0, 4, 2, 1),
    ]


@pytest.mark.parametrize(
    "m, n, num_cus, num_xcds",
    [
        (512, 256, 304, 8),  # fits in one wave
        (8192, 8192, 304, 8),  # 1024 tiles over several waves
        (6144, 3584, 304, 8),  # ragged edge tiles
        (4096, 4096, 80, 1),  # no XCD remap
    ],
)
def test_launch_wave_plan_covers_every_tile_once(m, n, num_cus, num_xcds):
    plan = build_launch_wave_signal_plan(m, n, _selector(num_xcds=num_xcds), num_cus)
    schedule = plan.schedule

    seen = []
    for batch in plan.batches:
        batch_tiles = set()
        for rect in batch.rects:
            assert rect.m_tile_start + rect.m_tile_count <= schedule.grid_m
            assert rect.n_tile_start + rect.n_tile_count <= schedule.grid_n
            assert not (batch_tiles & _rect_tiles(rect)), "rectangles in a batch overlap"
            batch_tiles |= _rect_tiles(rect)
        # The gate opens after exactly the tiles it releases, and those tiles report to it.
        assert batch.producers == len(batch_tiles)
        for tile_m, tile_n in batch_tiles:
            assert plan.signal_of_tile[tile_m * schedule.grid_n + tile_n] == batch.signal_id
        seen.extend(batch_tiles)

    assert len(seen) == schedule.num_tiles
    assert set(seen) == {(tm, tn) for tm in range(schedule.grid_m) for tn in range(schedule.grid_n)}


def test_launch_wave_plan_gates_follow_dispatch_waves():
    num_cus = 304
    plan = build_launch_wave_signal_plan(8192, 8192, _selector(), num_cus)
    schedule = plan.schedule

    expected_waves = -(-schedule.num_tiles // num_cus)
    assert plan.num_signals == expected_waves
    # Gate g collects the tiles dispatched at positions [g * num_cus, (g + 1) * num_cus).
    for position, tile in enumerate(schedule.tile_of_wg):
        assert plan.signal_of_tile[tile] == position // num_cus
    # Batches are queued in readiness order.
    assert [batch.signal_id for batch in plan.batches] == list(range(expected_waves))


def _schedule(m=8192, n=8192, **selector_kw):
    return tritonblas.schedule_for_selector(m, n, _selector(**selector_kw))


def test_plan_gates_one_gate_per_tile_follows_dispatch_order():
    schedule = _schedule(4096, 4096)
    batches = plan_gates(schedule, range(schedule.num_tiles))
    # Gate id is the linear tile id, so the queue is the order tiles are dispatched in.
    assert tuple(batch.signal_id for batch in batches) == schedule.tile_of_wg
    for batch in batches:
        assert batch.producers == 1
        assert sum(rect.m_tile_count * rect.n_tile_count for rect in batch.rects) == 1


def test_plan_gates_grouped_producers_and_queue_order():
    schedule = _schedule()
    group_size = 7
    signal_of_tile = [0] * schedule.num_tiles
    for position, tile in enumerate(schedule.tile_of_wg):
        signal_of_tile[tile] = position // group_size

    batches = plan_gates(schedule, signal_of_tile)
    num_gates = -(-schedule.num_tiles // group_size)
    assert [batch.signal_id for batch in batches] == list(range(num_gates))
    assert [batch.producers for batch in batches[:-1]] == [group_size] * (num_gates - 1)
    assert batches[-1].producers == (schedule.num_tiles % group_size or group_size)

    # Gates queue in the order their last tile is dispatched, whatever their ids are.
    inverse = schedule.wg_of_tile()
    last = {
        gate: max(inverse[tile] for tile, owner in enumerate(signal_of_tile) if owner == gate)
        for gate in range(num_gates)
    }
    shuffled = [(num_gates - 1) - gate for gate in signal_of_tile]  # same grouping, reversed ids
    queued = plan_gates(schedule, shuffled)
    assert [batch.signal_id for batch in queued] == [(num_gates - 1) - gate for gate in sorted(last, key=last.get)]
    assert [batch.producers for batch in queued] == [batch.producers for batch in batches]


def test_plan_gates_accepts_a_tensor_table():
    torch = pytest.importorskip("torch")
    schedule = _schedule(4096, 4096)
    table = torch.arange(schedule.num_tiles, dtype=torch.int32) // 3
    batches = plan_gates(schedule, table)
    assert sum(batch.producers for batch in batches) == schedule.num_tiles


def test_plan_gates_rejects_bad_maps():
    schedule = _schedule(4096, 4096)
    with pytest.raises(ValueError, match="tile grid"):
        plan_gates(schedule, [0])
    with pytest.raises(ValueError, match="non-negative"):
        plan_gates(schedule, [-1] * schedule.num_tiles)
    sparse = [0] * schedule.num_tiles
    sparse[-1] = 2
    with pytest.raises(ValueError, match="dense"):
        plan_gates(schedule, sparse)


def _row_schedule(tiles_m, tiles_n, tile_of_wg, num_xcds=1, group_m=1):
    return tritonblas.TileSchedule(
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        block_m=256,
        block_n=256,
        block_k=64,
        group_m=group_m,
        num_xcds=num_xcds,
        chunk_size=1,
        tile_of_wg=tuple(tile_of_wg),
    )


@pytest.mark.parametrize(
    "m, n, num_cus, expected",
    [
        # group_m=4, 256x256 tiles. 32x14 tiles: 304 // (4 * 14) = 5 groups per wave -> 20 rows.
        (8192, 3584, 304, 20),
        (16384, 3584, 304, 20),
        # A 2x1 grid: 76 groups per wave, clamped to the two tile rows.
        (512, 256, 304, 2),
    ],
)
def test_row_batches_default_size_is_the_existing_heuristic(m, n, num_cus, expected):
    schedule = selector_schedule(m, n, _selector())
    plan = plan_row_batches(schedule, num_cus)
    assert plan.m_tiles_per_batch == expected
    assert plan.num_batches == -(-schedule.grid_m // expected)
    assert sorted(plan.order) == list(range(plan.num_batches))


def test_row_batches_explicit_size_is_kept_and_clamped():
    schedule = selector_schedule(4096, 4096, _selector())
    assert plan_row_batches(schedule, 304, m_tiles_per_batch=3).m_tiles_per_batch == 3
    assert plan_row_batches(schedule, 304, m_tiles_per_batch=10**6).m_tiles_per_batch == schedule.grid_m
    with pytest.raises(ValueError, match="positive"):
        plan_row_batches(schedule, 304, m_tiles_per_batch=0)


def test_row_batches_queue_in_the_order_the_gemm_needs_them():
    # 4 tile rows x 2 columns, dispatched bottom row first: a producer that posts batches in
    # ascending M would make the first workgroups wait for the batch that is needed last.
    tiles_n = 2
    dispatch = [tile for row in (3, 2, 1, 0) for tile in (row * tiles_n, row * tiles_n + 1)]
    plan = plan_row_batches(_row_schedule(4, tiles_n, dispatch), num_cus=304, m_tiles_per_batch=1)
    assert plan.order == (3, 2, 1, 0)

    # Two rows per batch: batch 1 (rows 2-3) is needed first.
    plan = plan_row_batches(_row_schedule(4, tiles_n, dispatch), num_cus=304, m_tiles_per_batch=2)
    assert plan.order == (1, 0)


def test_row_batches_ascending_dispatch_keeps_ascending_order():
    schedule = _row_schedule(6, 2, range(12))
    assert plan_row_batches(schedule, num_cus=304, m_tiles_per_batch=2).order == (0, 1, 2)
