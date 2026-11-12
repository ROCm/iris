# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Copy-engine transfer batches built from the tile schedule tritonBLAS publishes.

tritonBLAS exposes the order its persistent GEMM visits output tiles through
``tritonblas.schedule_for_selector``. This module assigns one gate per hardware
launch wave, lets tritonBLAS derive each gate's producer count and readiness
order, and coalesces each gate's tiles into rectangular copies. Nothing here
re-derives the GEMM's tile permutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import tritonblas


@dataclass(frozen=True)
class TileRect:
    """A rectangle of output tiles copied by one 2D transfer."""

    m_tile_start: int
    n_tile_start: int
    m_tile_count: int
    n_tile_count: int


@dataclass(frozen=True)
class SignalBatch:
    """The transfers released once gate ``signal_id`` has received ``producers`` tile completions."""

    signal_id: int
    producers: int
    rects: tuple[TileRect, ...]


@dataclass(frozen=True)
class TileSignalPlan:
    """A tritonBLAS tile schedule with one gate per launch wave, plus the copies each gate releases."""

    schedule: tritonblas.TileSchedule
    num_cus: int
    #: In the order the consumer should queue them: each gate's final producer is dispatched no later
    #: than the next gate's, so a queue that blocks on each gate in turn never waits on a later one.
    batches: tuple[SignalBatch, ...]

    @property
    def num_signals(self) -> int:
        return self.schedule.num_signals

    @property
    def signal_of_tile(self) -> tuple[int, ...]:
        return self.schedule.signal_of_tile


def coalesce_tiles(tiles) -> tuple[TileRect, ...]:
    """Cover a set of ``(tile_m, tile_n)`` coordinates with rectangles, each tile exactly once.

    Contiguous runs along M within a tile column become one rectangle, and identical runs in
    adjacent columns are merged along N.
    """
    rows_by_col: dict[int, list[int]] = {}
    for tile_m, tile_n in tiles:
        rows_by_col.setdefault(tile_n, []).append(tile_m)

    rects: list[TileRect] = []
    # Run (m_start, m_count) -> index in ``rects`` of the rectangle that ends in the previous column.
    open_runs: dict[tuple[int, int], int] = {}
    for tile_n in sorted(rows_by_col):
        rows = sorted(rows_by_col[tile_n])
        runs = []
        run_start = prev = rows[0]
        for tile_m in rows[1:]:
            if tile_m != prev + 1:
                runs.append((run_start, prev - run_start + 1))
                run_start = tile_m
            prev = tile_m
        runs.append((run_start, prev - run_start + 1))

        next_open: dict[tuple[int, int], int] = {}
        for run in runs:
            idx = open_runs.get(run)
            if idx is not None and rects[idx].n_tile_start + rects[idx].n_tile_count == tile_n:
                rect = rects[idx]
                rects[idx] = TileRect(rect.m_tile_start, rect.n_tile_start, rect.m_tile_count, rect.n_tile_count + 1)
            else:
                rects.append(TileRect(run[0], tile_n, run[1], 1))
                idx = len(rects) - 1
            next_open[run] = idx
        open_runs = next_open

    return tuple(rects)


def build_launch_wave_signal_plan(m: int, n: int, selector, num_cus: int) -> TileSignalPlan:
    """Plan one gate per launch wave of ``num_cus`` workgroups for the GEMM ``selector`` will launch.

    ``m`` and ``n`` are the GEMM output dimensions on this rank.
    """
    schedule = tritonblas.schedule_for_selector(m, n, selector)
    if schedule is None:
        raise ValueError("the selected GEMM has no static tile schedule to signal from")
    # Waves are numbered from 1 and dense, so they map directly onto gate ids.
    schedule = schedule.with_signal_layout([wave - 1 for wave in schedule.ready_wave_of_tile(num_cus)])
    batches = tuple(
        SignalBatch(
            signal_id=arm.signal_id,
            producers=arm.producers,
            rects=_coalesce_by_band(arm.tiles, schedule.group_m),
        )
        for arm in schedule.signal_plan.arms
    )
    return TileSignalPlan(schedule=schedule, num_cus=num_cus, batches=batches)


def _coalesce_by_band(tiles, band_m: int) -> tuple[TileRect, ...]:
    """Coalesce each ``band_m``-row band separately.

    The GEMM swizzles tiles within bands of ``GROUP_M`` rows, so a wave's tiles form regular shapes
    inside each band. A run that crossed a band boundary would differ from its neighbours' runs and
    stop them merging along N.
    """
    bands: dict[int, list[tuple[int, int]]] = {}
    for tile in tiles:
        bands.setdefault(tile[0] // band_m, []).append(tile)
    return tuple(rect for band in sorted(bands) for rect in coalesce_tiles(bands[band]))
