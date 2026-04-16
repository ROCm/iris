# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Helpers for describing the tritonBLAS launch-wave tile schedule.

The current tritonBLAS path launches one program per output tile and lets the
hardware schedule those programs in waves of active CUs. The helper in this
module mirrors that launch order, including the chunked XCD remap, and
coalesces the tiles of each hardware wave into one or more rectangular
transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def chiplet_transform_chunked(pid: int, num_workgroups: int, num_xcds: int, chunk_size: int) -> int:
    if num_xcds <= 1 or chunk_size <= 0:
        return pid
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    xcd = pid % num_xcds
    return chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk


def default_chunk_size(total_tiles: int, group_size_m: int, num_xcds: int) -> int:
    chunk_size = group_size_m * group_size_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    return max(1, chunk_size)


def grouped_tile_coords(tile_id: int, num_tiles_m: int, num_tiles_n: int, group_size_m: int) -> tuple[int, int, int]:
    num_pid_in_group = group_size_m * num_tiles_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    actual_group_size_m = min(num_tiles_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % actual_group_size_m)
    pid_n = (tile_id % num_pid_in_group) // actual_group_size_m
    return pid_m, pid_n, group_id


@dataclass(frozen=True)
class LaunchWaveTransfer:
    wave_id: int
    group_id: int
    m_tile_start: int
    n_tile_start: int
    m_tile_count: int
    n_tile_count: int
    tile_count: int


@dataclass(frozen=True)
class LaunchWavePlan:
    num_tiles_m: int
    num_tiles_n: int
    total_tiles: int
    launch_grid: int
    wave_size: int
    num_xcds: int
    chunk_size: int
    num_waves: int
    wave_tile_counts: tuple[int, ...]
    transfers: tuple[LaunchWaveTransfer, ...]


def _coalesce_group_columns(
    wave_id: int,
    group_id: int,
    first_pid_m: int,
    columns: dict[int, set[int]],
) -> list[LaunchWaveTransfer]:
    transfers: list[LaunchWaveTransfer] = []
    merged: list[tuple[int, int, int, int]] = []

    for pid_n in sorted(columns):
        local_ms = sorted(columns[pid_n])
        if not local_ms:
            continue

        seg_start = local_ms[0]
        seg_prev = local_ms[0]
        for local_m in local_ms[1:]:
            if local_m == seg_prev + 1:
                seg_prev = local_m
                continue
            merged.append((pid_n, seg_start, seg_prev - seg_start + 1, 1))
            seg_start = local_m
            seg_prev = local_m
        merged.append((pid_n, seg_start, seg_prev - seg_start + 1, 1))

    for pid_n, local_m_start, m_tile_count, n_tile_count in merged:
        if transfers:
            prev = transfers[-1]
            if (
                prev.group_id == group_id
                and prev.n_tile_start + prev.n_tile_count == pid_n
                and prev.m_tile_start == first_pid_m + local_m_start
                and prev.m_tile_count == m_tile_count
            ):
                transfers[-1] = LaunchWaveTransfer(
                    wave_id=prev.wave_id,
                    group_id=prev.group_id,
                    m_tile_start=prev.m_tile_start,
                    n_tile_start=prev.n_tile_start,
                    m_tile_count=prev.m_tile_count,
                    n_tile_count=prev.n_tile_count + n_tile_count,
                    tile_count=prev.tile_count + m_tile_count * n_tile_count,
                )
                continue

        transfers.append(
            LaunchWaveTransfer(
                wave_id=wave_id,
                group_id=group_id,
                m_tile_start=first_pid_m + local_m_start,
                n_tile_start=pid_n,
                m_tile_count=m_tile_count,
                n_tile_count=n_tile_count,
                tile_count=m_tile_count * n_tile_count,
            )
        )

    return transfers


def _coalesce_group_rows(
    wave_id: int,
    group_id: int,
    first_pid_m: int,
    columns: dict[int, set[int]],
) -> list[LaunchWaveTransfer]:
    rows: dict[int, set[int]] = {}
    for pid_n, local_ms in columns.items():
        for local_m in local_ms:
            rows.setdefault(local_m, set()).add(pid_n)

    transfers: list[LaunchWaveTransfer] = []
    merged: list[tuple[int, int, int, int]] = []

    for local_m in sorted(rows):
        ns = sorted(rows[local_m])
        if not ns:
            continue

        seg_start = ns[0]
        seg_prev = ns[0]
        for pid_n in ns[1:]:
            if pid_n == seg_prev + 1:
                seg_prev = pid_n
                continue
            merged.append((local_m, seg_start, 1, seg_prev - seg_start + 1))
            seg_start = pid_n
            seg_prev = pid_n
        merged.append((local_m, seg_start, 1, seg_prev - seg_start + 1))

    for local_m_start, n_tile_start, m_tile_count, n_tile_count in merged:
        if transfers:
            prev = transfers[-1]
            if (
                prev.group_id == group_id
                and prev.m_tile_start + prev.m_tile_count == first_pid_m + local_m_start
                and prev.n_tile_start == n_tile_start
                and prev.n_tile_count == n_tile_count
            ):
                transfers[-1] = LaunchWaveTransfer(
                    wave_id=prev.wave_id,
                    group_id=prev.group_id,
                    m_tile_start=prev.m_tile_start,
                    n_tile_start=prev.n_tile_start,
                    m_tile_count=prev.m_tile_count + m_tile_count,
                    n_tile_count=prev.n_tile_count,
                    tile_count=prev.tile_count + m_tile_count * n_tile_count,
                )
                continue

        transfers.append(
            LaunchWaveTransfer(
                wave_id=wave_id,
                group_id=group_id,
                m_tile_start=first_pid_m + local_m_start,
                n_tile_start=n_tile_start,
                m_tile_count=m_tile_count,
                n_tile_count=n_tile_count,
                tile_count=m_tile_count * n_tile_count,
            )
        )

    return transfers


def build_launch_wave_plan(
    num_tiles_m: int,
    num_tiles_n: int,
    group_size_m: int,
    launch_grid: int,
    wave_size: int,
    num_xcds: int,
    chunk_size: int | None = None,
    merge_order: str = "column",
) -> LaunchWavePlan:
    total_tiles = num_tiles_m * num_tiles_n
    if launch_grid <= 0:
        raise ValueError("launch_grid must be positive")
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    if total_tiles <= 0:
        raise ValueError("tile grid must be non-empty")
    if merge_order not in {"column", "row"}:
        raise ValueError("merge_order must be 'column' or 'row'")

    if chunk_size is None:
        chunk_size = default_chunk_size(total_tiles, group_size_m, num_xcds)

    launch_grid = max(launch_grid, total_tiles)
    num_waves = ceil_div(launch_grid, wave_size)
    wave_tile_counts: list[int] = []
    transfers: list[LaunchWaveTransfer] = []

    for wave_id in range(num_waves):
        pid_start = wave_id * wave_size
        pid_end = min(pid_start + wave_size, launch_grid)

        groups: dict[int, tuple[int, dict[int, set[int]]]] = {}
        tiles_in_wave = 0

        for pid in range(pid_start, pid_end):
            tile_id = chiplet_transform_chunked(pid, launch_grid, num_xcds, chunk_size)
            if tile_id >= total_tiles:
                continue

            pid_m, pid_n, group_id = grouped_tile_coords(tile_id, num_tiles_m, num_tiles_n, group_size_m)
            first_pid_m = group_id * group_size_m
            columns = groups.setdefault(group_id, (first_pid_m, {}))[1]
            local_m = pid_m - first_pid_m
            columns.setdefault(pid_n, set()).add(local_m)
            tiles_in_wave += 1

        wave_tile_counts.append(tiles_in_wave)
        if tiles_in_wave == 0:
            continue

        for group_id in sorted(groups):
            first_pid_m, columns = groups[group_id]
            if merge_order == "row":
                transfers.extend(_coalesce_group_rows(wave_id, group_id, first_pid_m, columns))
            else:
                transfers.extend(_coalesce_group_columns(wave_id, group_id, first_pid_m, columns))

    return LaunchWavePlan(
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        total_tiles=total_tiles,
        launch_grid=launch_grid,
        wave_size=wave_size,
        num_xcds=num_xcds,
        chunk_size=chunk_size,
        num_waves=num_waves,
        wave_tile_counts=tuple(wave_tile_counts),
        transfers=tuple(transfers),
    )


def build_launch_wave_plan_for_shape(
    m: int,
    n: int,
    block_m: int,
    block_n: int,
    group_size_m: int,
    launch_grid: int,
    wave_size: int,
    num_xcds: int,
    chunk_size: int | None = None,
    merge_order: str = "column",
) -> LaunchWavePlan:
    num_tiles_m = ceil_div(m, block_m)
    num_tiles_n = ceil_div(n, block_n)
    return build_launch_wave_plan(
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        group_size_m=group_size_m,
        launch_grid=launch_grid,
        wave_size=wave_size,
        num_xcds=num_xcds,
        chunk_size=chunk_size,
        merge_order=merge_order,
    )
