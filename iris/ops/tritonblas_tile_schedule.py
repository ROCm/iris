# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Copy-engine transfer batches built from the tile schedule tritonBLAS publishes.

tritonBLAS publishes the order its persistent GEMM visits output tiles
(``tritonblas.schedule_for_selector``) and a wave model for when each tile retires.
That is all it says about the GEMM. This module owns the signalling policy built on it:
which gate each tile reports to, how many producers a gate waits for, the order the
copy engine queues the gates in, and how a gate's tiles become rectangular copies.
Nothing here re-derives the GEMM's tile permutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
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
    #: Gate id written by each output tile, indexed by row-major linear tile id. This is the table
    #: the GEMM epilogue reads (``EpilogueConfig(map_type="table", tile_to_slot=...)``).
    signal_of_tile: tuple[int, ...]
    #: In the order the consumer should queue them: each gate's final producer is dispatched no later
    #: than the next gate's, so a queue that blocks on each gate in turn never waits on a later one.
    batches: tuple[SignalBatch, ...]

    @property
    def num_signals(self) -> int:
        return len(self.batches)


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


def selector_wave_size(selector, device: torch.device) -> int:
    """Workgroups the GEMM runs at once: the selector's active CU count, else the device's."""
    wave_size = getattr(selector, "_ACTIVE_CU", None)
    if wave_size is None or wave_size <= 0:
        wave_size = torch.cuda.get_device_properties(device).multi_processor_count
    return int(wave_size)


def selector_schedule(m: int, n: int, selector) -> tritonblas.TileSchedule:
    """The tile schedule of the GEMM ``selector`` will launch on an ``m`` x ``n`` output."""
    schedule = tritonblas.schedule_for_selector(m, n, selector)
    if schedule is None:
        raise ValueError("the selected GEMM has no static tile schedule to plan from")
    return schedule


def plan_gates(schedule: tritonblas.TileSchedule, signal_of_tile) -> tuple[SignalBatch, ...]:
    """Derive each gate's producer count, tiles and queue position from a tile-to-gate map.

    ``signal_of_tile[t]`` is the gate that row-major linear tile ``t`` increments. Gate ids must be
    dense (``0..num_gates-1``). A gate opens once all of its tiles have been produced, so gates are
    ordered by when their last tile is dispatched: a queue that blocks on each gate in turn then
    never waits behind a gate that opens later.
    """
    if hasattr(signal_of_tile, "tolist"):
        signal_of_tile = signal_of_tile.tolist()
    signal_of_tile = [int(signal) for signal in signal_of_tile]
    if len(signal_of_tile) != schedule.num_tiles:
        raise ValueError(f"signal_of_tile has {len(signal_of_tile)} entries; the tile grid has {schedule.num_tiles}")
    if not signal_of_tile or min(signal_of_tile) < 0:
        raise ValueError("signal ids must be non-negative")
    num_gates = max(signal_of_tile) + 1
    if set(signal_of_tile) != set(range(num_gates)):
        raise ValueError("signal ids must be dense from 0 through num_gates - 1")

    dispatch_position = schedule.wg_of_tile()
    tiles_of_gate: list[list[tuple[int, int]]] = [[] for _ in range(num_gates)]
    last_dispatched = [-1] * num_gates
    for tile, gate in enumerate(signal_of_tile):
        tiles_of_gate[gate].append((tile // schedule.grid_n, tile % schedule.grid_n))
        last_dispatched[gate] = max(last_dispatched[gate], dispatch_position[tile])
    return tuple(
        SignalBatch(
            signal_id=gate,
            producers=len(tiles_of_gate[gate]),
            rects=_coalesce_by_band(tiles_of_gate[gate], schedule.group_m),
        )
        for gate in sorted(range(num_gates), key=last_dispatched.__getitem__)
    )


def build_launch_wave_signal_plan(m: int, n: int, selector, num_cus: int) -> TileSignalPlan:
    """Plan one gate per launch wave of ``num_cus`` workgroups for the GEMM ``selector`` will launch.

    ``m`` and ``n`` are the GEMM output dimensions on this rank.
    """
    schedule = selector_schedule(m, n, selector)
    # Waves are numbered from 1 and dense, so they map directly onto gate ids.
    signal_of_tile = tuple(wave - 1 for wave in schedule.ready_wave_of_tile(num_cus))
    return TileSignalPlan(
        schedule=schedule,
        num_cus=num_cus,
        signal_of_tile=signal_of_tile,
        batches=plan_gates(schedule, signal_of_tile),
    )


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


@dataclass(frozen=True)
class RowBatchPlan:
    """M-row batches a GEMM consumes, for a producer that delivers A in bands of whole M-tile rows.

    Batch ``b`` holds M-tile rows ``[b * m_tiles_per_batch, (b + 1) * m_tiles_per_batch)`` and is
    announced by one flag. Every GEMM tile in those rows waits on that flag.
    """

    schedule: tritonblas.TileSchedule
    num_cus: int
    m_tiles_per_batch: int
    #: Batch ids in the order the GEMM first needs them. A producer that handles batches in one queue
    #: should post them in this order, so a batch needed early never waits behind one needed later.
    order: tuple[int, ...]

    @property
    def num_batches(self) -> int:
        return len(self.order)


def plan_row_batches(
    schedule: tritonblas.TileSchedule, num_cus: int, m_tiles_per_batch: int | None = None
) -> RowBatchPlan:
    """Plan the M-row batches a GEMM with this ``schedule`` consumes.

    The batch order is a fact read from the schedule: batches are ordered by the dispatch position
    of the first tile that needs them, which is not ascending in M in general.

    The default batch size is the formula this op used before it read the schedule, not something
    the schedule determines: the M-tile rows that ``num_cus`` workgroups cover if tiles were
    dispatched in row-major order, in whole ``GROUP_M`` bands. It has not been re-tuned. Sizing
    batches by the rows the first launch wave really touches (more than the formula gives, since
    the GEMM's chiplet remapping spreads the first wave over more rows) was measured 4-6% slower
    at M=4096 and 8192 on 8 ranks (N=3584, K=8192), and equal at M=16384 and 32768: a larger first
    batch delays the first tiles more than fewer transfers save. Other batch sizes were not swept.
    """
    grid_m, grid_n = schedule.grid_m, schedule.grid_n
    if m_tiles_per_batch is None:
        groups_per_wave = max(1, num_cus // max(1, schedule.group_m * grid_n))
        m_tiles_per_batch = max(1, groups_per_wave * schedule.group_m)
    if m_tiles_per_batch < 1:
        raise ValueError(f"m_tiles_per_batch must be positive, got {m_tiles_per_batch}")
    m_tiles_per_batch = min(m_tiles_per_batch, grid_m)

    dispatch_position = schedule.wg_of_tile()
    num_batches = -(-grid_m // m_tiles_per_batch)
    first_needed = [
        min(
            dispatch_position[row * grid_n + col]
            for row in range(batch * m_tiles_per_batch, min((batch + 1) * m_tiles_per_batch, grid_m))
            for col in range(grid_n)
        )
        for batch in range(num_batches)
    ]
    order = tuple(sorted(range(num_batches), key=first_needed.__getitem__))
    return RowBatchPlan(schedule=schedule, num_cus=num_cus, m_tiles_per_batch=m_tiles_per_batch, order=order)
