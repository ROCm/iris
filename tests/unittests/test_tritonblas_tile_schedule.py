# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Copy-engine signal batches built from tritonBLAS's published tile schedule (host only).

from types import SimpleNamespace

import pytest

from iris.ops.tritonblas_tile_schedule import TileRect, build_launch_wave_signal_plan, coalesce_tiles


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
