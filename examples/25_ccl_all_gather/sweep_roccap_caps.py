#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Sweep iris CCL all_gather roccap captures: run torchrun + roccap_wrapper for each
parameter tuple, then rename generated outputs to matching stems:

  {stem}.cap
  {stem}_heap_bases.json  (from iris_rank_{r}_allocator_views.json)

where stem is:

  {kernel}_{m}x{n}_{bm}x{bn}_{sms}sms_{stage}stage_{datatype}_{cm}_{warps}warps_{nproc}nproc_rank{r}

  {kernel} is the value of ``--kernel`` (roccap / dispatch name). {cm} is legacy | none | wt | cs
  (from --cache-modifier). ``--use_inline`` forwards to example.py (mutually exclusive with
  ``--use_gluon``).

Run from this directory (same as example.py):

  python sweep_roccap_caps.py \\
    --kernel persistent_all_gather_gluon_gfx1250 \\
    -m 512 --n 256 \\
    --block_size_m 32 --block_size_n 64 \\
    --comm_sms 64 --num_stages 1 --num_warps 4 \\
    --datatype fp32 \\
    --use_gluon \\
    --cache-modifier legacy .wt .cs

  scp persistent_all_gather_gluon_1024x1024_128x128_64sms_1stage_fp32_cs_32warps_8nproc_rank0* jonathou@atlvscode0002.
amd.com:/proj/SPG_data_vault/msam_workspace/workloads/iris/mi450/all_gather/lds/temp/.

Pass multiple values to any knob to take the Cartesian product (e.g. --num_stages 1 2 4).
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_int_list(name: str, values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        try:
            out.append(int(v))
        except ValueError as e:
            raise SystemExit(f"{name} expects integers, got {v!r}") from e
    return out


def _cache_modifier_stem(cm: str) -> str:
    """Short token for output filenames (matches example.py --cache-modifier values)."""
    if cm == ".wt":
        return "wt"
    if cm == ".cs":
        return "cs"
    return cm


def _kernel_stem(kernel: str) -> str:
    """Sanitize --kernel for use in filenames."""
    return kernel.replace("/", "_").replace("\\", "_")


def cap_basename(
    kernel: str,
    m: int,
    n: int,
    block_m: int,
    block_n: int,
    comm_sms: int,
    num_stages: int,
    datatype: str,
    cache_modifier: str,
    num_warps: int,
    nproc: int,
    rank: int,
) -> str:
    k = _kernel_stem(kernel)
    cm = _cache_modifier_stem(cache_modifier)
    return (
        f"{k}_{m}x{n}_"
        f"{block_m}x{block_n}_"
        f"{comm_sms}sms_{num_stages}stage_{datatype}_{cm}_"
        f"{num_warps}warps_{nproc}nproc_rank{rank}"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    default_wrapper = here.parent.parent / "scripts" / "roccap_wrapper.py"

    p = argparse.ArgumentParser(
        description="Sweep roccap captures for CCL all_gather with renamed .cap and heap_bases .json outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--kernel",
        "-k",
        default="persistent_all_gather",
        help="Kernel name for roccap -k / dispatch filter",
    )
    p.add_argument(
        "--wrapper",
        type=Path,
        default=default_wrapper,
        help="Path to roccap_wrapper.py",
    )
    p.add_argument(
        "--example",
        type=Path,
        default=here / "example.py",
        help="Example script passed to the wrapper",
    )
    p.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="torchrun local world size",
    )
    p.add_argument("-m", nargs="+", required=True, help="Row count per rank (one or more values)")
    p.add_argument(
        "-n",
        "--n-cols",
        dest="n_cols",
        nargs="+",
        required=True,
        help="Column count (one or more values)",
    )
    p.add_argument("--block_size_m", nargs="+", required=True)
    p.add_argument("--block_size_n", nargs="+", required=True)
    p.add_argument("--comm_sms", nargs="+", required=True)
    p.add_argument("--num_stages", nargs="+", required=True)
    p.add_argument("--num_warps", nargs="+", required=True)
    p.add_argument(
        "--datatype",
        nargs="+",
        default=["fp16"],
        choices=["fp16", "fp32", "bf16"],
        metavar="DT",
        help="Data type(s); forwarded to example.py (default: fp16)",
    )
    p.add_argument(
        "--use_gluon",
        action="store_true",
        help="Forward --use_gluon to example.py",
    )
    p.add_argument(
        "--use_inline",
        action="store_true",
        help="Forward --use_inline to example.py (persistent_all_gather_inline; not with --use_gluon)",
    )
    p.add_argument(
        "--cache-modifier",
        nargs="+",
        default=["legacy"],
        choices=["legacy", "none", ".wt", ".cs"],
        metavar="CM",
        help="Forwarded to example.py --cache-modifier (one or more; Cartesian product)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and renames only",
    )
    args = p.parse_args()

    if args.use_gluon and args.use_inline:
        sys.exit("Cannot combine --use_gluon and --use_inline")

    m_list = _parse_int_list("-m", args.m)
    n_list = _parse_int_list("-n / --n-cols", args.n_cols)
    bm_list = _parse_int_list("--block_size_m", args.block_size_m)
    bn_list = _parse_int_list("--block_size_n", args.block_size_n)
    sms_list = _parse_int_list("--comm_sms", args.comm_sms)
    stages_list = _parse_int_list("--num_stages", args.num_stages)
    warps_list = _parse_int_list("--num_warps", args.num_warps)
    dtype_list = list(args.datatype)
    cache_modifier_list = list(args.cache_modifier)

    wrapper = args.wrapper.resolve()
    example = args.example.resolve()
    if not wrapper.is_file():
        sys.exit(f"Wrapper not found: {wrapper}")
    if not example.is_file():
        sys.exit(f"Example script not found: {example}")

    kernel = args.kernel
    nproc = args.nproc_per_node
    combos = list(
        itertools.product(
            m_list,
            n_list,
            bm_list,
            bn_list,
            sms_list,
            stages_list,
            dtype_list,
            warps_list,
            cache_modifier_list,
        )
    )
    print(f"Total configurations: {len(combos)}", file=sys.stderr)

    for m, n, bm, bn, sms, stages, dtype, warps, cache_mod in combos:
        child_args: list[str] = [
            str(example),
            "-m",
            str(m),
            "-n",
            str(n),
            "--block_size_m",
            str(bm),
            "--block_size_n",
            str(bn),
            "--comm_sms",
            str(sms),
            "--num_stages",
            str(stages),
            "--num_warps",
            str(warps),
            "--datatype",
            dtype,
            "--cache-modifier",
            cache_mod,
        ]
        if args.use_gluon:
            child_args.append("--use_gluon")
        if args.use_inline:
            child_args.append("--use_inline")

        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            "--standalone",
            str(wrapper),
            "-k",
            kernel,
        ] + child_args

        print("\n" + " ".join(cmd), file=sys.stderr)
        if args.dry_run:
            for r in range(args.nproc_per_node):
                stem = cap_basename(kernel, m, n, bm, bn, sms, stages, dtype, cache_mod, warps, nproc, r)
                old_cap = here / f"{kernel}_rank_{r}.cap"
                new_cap = here / f"{stem}.cap"
                old_json = here / f"iris_rank_{r}_allocator_views.json"
                new_json = here / f"{stem}_heap_bases.json"
                print(f"  would rename {old_cap.name} -> {new_cap.name}", file=sys.stderr)
                print(f"  would rename {old_json.name} -> {new_json.name}", file=sys.stderr)
            continue

        env = os.environ.copy()
        proc = subprocess.run(cmd, cwd=here, env=env)
        if proc.returncode != 0:
            sys.exit(f"torchrun failed with exit code {proc.returncode} for m={m} n={n} cache_mod={cache_mod!r} ...")

        for r in range(args.nproc_per_node):
            stem = cap_basename(kernel, m, n, bm, bn, sms, stages, dtype, cache_mod, warps, nproc, r)
            old_cap = here / f"{kernel}_rank_{r}.cap"
            new_cap = here / f"{stem}.cap"
            old_json = here / f"iris_rank_{r}_allocator_views.json"
            new_json = here / f"{stem}_heap_bases.json"

            if not old_cap.is_file():
                print(
                    f"Warning: expected cap not found (rank {r}): {old_cap}",
                    file=sys.stderr,
                )
            else:
                if new_cap.exists():
                    sys.exit(f"Refusing to overwrite existing file: {new_cap}")
                shutil.move(str(old_cap), str(new_cap))
                print(f"  wrote {new_cap.name}", file=sys.stderr)

            if not old_json.is_file():
                print(
                    f"Warning: expected iris allocator_views json not found (rank {r}): {old_json}",
                    file=sys.stderr,
                )
            else:
                if new_json.exists():
                    sys.exit(f"Refusing to overwrite existing file: {new_json}")
                shutil.move(str(old_json), str(new_json))
                print(f"  wrote {new_json.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
