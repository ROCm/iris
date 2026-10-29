# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Compact XCD-local tile schedule for the two-shot GEMM + all-reduce."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class CompactXcdSchedule:
    """Persistent GEMM tile ownership expressed as dense XCD-local slots."""

    num_tiles_m: int
    num_tiles_n: int
    total_tiles: int
    launch_grid: int
    sdma_wave_size: int
    num_xcds: int
    programs_per_xcd: int
    slots_per_xcd: int
    num_waves: int
    chunk_size: int
    tiles_by_xcd_wave: tuple[tuple[tuple[int, ...], ...], ...]


def build_compact_xcd_schedule(
    num_tiles_m: int,
    num_tiles_n: int,
    group_size_m: int,
    requested_num_sms: int,
    num_xcds: int,
) -> CompactXcdSchedule:
    """Mirror TritonBLAS's full tile grid and partition it into SDMA-ready waves."""
    total_tiles = num_tiles_m * num_tiles_n
    if total_tiles <= 0:
        raise ValueError("tile grid must be non-empty")
    if requested_num_sms <= 0:
        raise ValueError("requested_num_sms must be positive")
    if num_xcds <= 0:
        raise ValueError("num_xcds must be positive")

    # TritonBLAS launches one program per logical tile. SDMA readiness uses a
    # separate, hardware-sized program wave so transfer can overlap later GEMM.
    launch_grid = total_tiles
    sdma_wave_size = ceil_div(min(requested_num_sms, total_tiles), num_xcds) * num_xcds
    programs_per_xcd = sdma_wave_size // num_xcds
    num_waves = ceil_div(launch_grid, sdma_wave_size)
    slots_per_xcd = num_waves * programs_per_xcd
    chunk_size = min(group_size_m * group_size_m, max(1, total_tiles // num_xcds))
    tiles: list[list[list[int]]] = [[[] for _ in range(num_waves)] for _ in range(num_xcds)]
    slots: list[list[list[int]]] = [[[] for _ in range(num_waves)] for _ in range(num_xcds)]

    for raw_pid in range(launch_grid):
        xcd_id = raw_pid % num_xcds
        local_program = raw_pid // num_xcds
        first_tile = chiplet_transform_chunked(raw_pid, launch_grid, num_xcds, chunk_size)
        if first_tile >= total_tiles:
            continue
        wave_id = raw_pid // sdma_wave_size
        program_in_wave = local_program % programs_per_xcd
        tiles[xcd_id][wave_id].append(first_tile)
        slots[xcd_id][wave_id].append(wave_id * programs_per_xcd + program_in_wave)

    flattened_tiles = [tile_id for xcd in tiles for wave in xcd for tile_id in wave]
    if sorted(flattened_tiles) != list(range(total_tiles)):
        raise ValueError("compact XCD schedule must assign each tile exactly once")
    for xcd_id, xcd_slots in enumerate(slots):
        for wave_id, wave_slots in enumerate(xcd_slots):
            expected_slots = list(range(wave_id * programs_per_xcd, wave_id * programs_per_xcd + len(wave_slots)))
            if sorted(wave_slots) != expected_slots:
                raise ValueError(f"XCD {xcd_id} wave {wave_id} does not occupy a dense compact slot run")

    return CompactXcdSchedule(
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        total_tiles=total_tiles,
        launch_grid=launch_grid,
        sdma_wave_size=sdma_wave_size,
        num_xcds=num_xcds,
        programs_per_xcd=programs_per_xcd,
        slots_per_xcd=slots_per_xcd,
        num_waves=num_waves,
        chunk_size=chunk_size,
        tiles_by_xcd_wave=tuple(tuple(tuple(wave) for wave in xcd) for xcd in tiles),
    )
