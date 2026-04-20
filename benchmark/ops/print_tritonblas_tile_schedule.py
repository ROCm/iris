#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Print the tritonBLAS launch-wave GEMM tile schedule.

This mirrors the current tritonBLAS matmul path used by the copy-engine code:
1. launch one workgroup per output tile
2. remap program IDs with chiplet_transform_chunked()
3. let hardware schedule those workgroups in waves of active CUs
4. map remapped tile IDs to (pid_m, pid_n) using GROUP_SIZE_M swizzling

The output is grouped into hardware waves of `wave_size` workgroups so it is
easy to see which tile coordinates are active in the first 304 WGs, second
304 WGs, and so on.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

_SCRIPT_PATH = Path(__file__).resolve()
_HELPER_PATH = None
for _parent in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    _candidate = _parent / "iris" / "ops" / "tritonblas_launch_wave_schedule.py"
    if _candidate.is_file():
        _HELPER_PATH = _candidate
        break
if _HELPER_PATH is None:
    raise FileNotFoundError(
        f"Unable to locate iris/ops/tritonblas_launch_wave_schedule.py starting from {_SCRIPT_PATH}"
    )

_SPEC = importlib.util.spec_from_file_location("tritonblas_launch_wave_schedule", str(_HELPER_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load helper module from {_HELPER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_launch_wave_plan = _MODULE.build_launch_wave_plan
ceil_div = _MODULE.ceil_div
chiplet_transform_chunked = _MODULE.chiplet_transform_chunked
default_chunk_size = _MODULE.default_chunk_size
grouped_tile_coords = _MODULE.grouped_tile_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print tritonBLAS XCD-aware launch-wave GEMM tile schedule by iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m", type=int, required=True, help="Problem M dimension")
    parser.add_argument("--n", type=int, required=True, help="Problem N dimension")
    parser.add_argument("--block-m", type=int, required=True, help="BLOCK_SIZE_M")
    parser.add_argument("--block-n", type=int, required=True, help="BLOCK_SIZE_N")
    parser.add_argument("--group-size-m", type=int, required=True, help="GROUP_SIZE_M")
    parser.add_argument("--wave-size", type=int, default=304, help="Active workgroups / CUs per hardware wave")
    parser.add_argument("--num-xcds", type=int, default=8, help="Number of XCDs used by the transform")
    parser.add_argument(
        "--merge-order",
        type=str,
        choices=("column", "row"),
        default="column",
        help="How to coalesce neighboring tiles into transfer rectangles.",
    )
    parser.add_argument(
        "--launch-grid",
        type=int,
        default=None,
        help="Optional explicit kernel launch grid. Defaults to total output tiles.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional explicit XCD chunk size. Defaults to min(group_size_m^2, total_tiles // num_xcds).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="How many waves of workgroups to print. Defaults to all waves.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the compact per-iteration summaries, not every workgroup entry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_tiles_m = ceil_div(args.m, args.block_m)
    num_tiles_n = ceil_div(args.n, args.block_n)
    total_tiles = num_tiles_m * num_tiles_n
    launch_grid = total_tiles if args.launch_grid is None else args.launch_grid

    if args.chunk_size is None:
        chunk_size = default_chunk_size(total_tiles, args.group_size_m, args.num_xcds)
    else:
        chunk_size = args.chunk_size

    plan = build_launch_wave_plan(
        num_tiles_m=num_tiles_m,
        num_tiles_n=num_tiles_n,
        group_size_m=args.group_size_m,
        launch_grid=launch_grid,
        wave_size=args.wave_size,
        num_xcds=args.num_xcds,
        chunk_size=chunk_size,
        merge_order=args.merge_order,
    )

    max_iterations = plan.num_waves
    iterations = max_iterations if args.iterations is None else min(args.iterations, max_iterations)

    print(f"Shape          : M={args.m} N={args.n}")
    print(f"Tile shape     : BLOCK_M={args.block_m} BLOCK_N={args.block_n}")
    print(f"Tile grid      : num_tiles_m={num_tiles_m} num_tiles_n={num_tiles_n} total_tiles={total_tiles}")
    print(f"Group size     : group_size_m={args.group_size_m} tiles_per_group={args.group_size_m * num_tiles_n}")
    print(f"Hardware       : wave_size={args.wave_size} num_xcds={args.num_xcds} chunk_size={chunk_size}")
    print(f"Merge order    : {args.merge_order}")
    print(f"Launch grid    : {launch_grid}")
    print(f"Iterations     : printing {iterations} / {max_iterations}")
    print(f"Transfers      : {len(plan.transfers)}")
    print()

    for iteration in range(iterations):
        entries: list[tuple[int, int, int, int, int, int]] = []
        iteration_transfers = [transfer for transfer in plan.transfers if transfer.wave_id == iteration]
        pid_start = iteration * args.wave_size
        pid_end = min(pid_start + args.wave_size, launch_grid)
        for pid in range(pid_start, pid_end):
            transformed_pid = chiplet_transform_chunked(pid, launch_grid, args.num_xcds, chunk_size)
            tile_id = transformed_pid
            if tile_id >= total_tiles:
                continue
            pid_m, pid_n, group_id = grouped_tile_coords(tile_id, num_tiles_m, num_tiles_n, args.group_size_m)
            xcd = pid % args.num_xcds if args.num_xcds > 0 else 0
            entries.append((pid, xcd, transformed_pid, tile_id, pid_m, pid_n, group_id))

        if not entries:
            break

        unique_groups = sorted({entry[6] for entry in entries})
        m_min = min(entry[4] for entry in entries)
        m_max = max(entry[4] for entry in entries)
        n_min = min(entry[5] for entry in entries)
        n_max = max(entry[5] for entry in entries)
        print(
            f"Iteration {iteration:2d}: {len(entries):3d} tiles  "
            f"groups={unique_groups}  m=[{m_min},{m_max}]  n=[{n_min},{n_max}]  "
            f"transfers={len(iteration_transfers)}"
        )
        for transfer in iteration_transfers:
            print(
                f"    transfer  group={transfer.group_id:2d}  "
                f"m=[{transfer.m_tile_start},{transfer.m_tile_start + transfer.m_tile_count - 1}]  "
                f"n=[{transfer.n_tile_start},{transfer.n_tile_start + transfer.n_tile_count - 1}]  "
                f"shape={transfer.m_tile_count}x{transfer.n_tile_count}"
            )

        if args.summary_only:
            continue

        print("  pid  xcd  remap  tile_id   m   n  group")
        for pid, xcd, transformed_pid, tile_id, pid_m, pid_n, group_id in entries:
            print(f"  {pid:3d}  {xcd:3d}  {transformed_pid:5d}  {tile_id:7d}  {pid_m:2d}  {pid_n:2d}  {group_id:5d}")
        print()


if __name__ == "__main__":
    main()
