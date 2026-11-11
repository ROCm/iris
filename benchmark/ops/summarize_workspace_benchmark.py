# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Compare matched base/head benchmark reports without requiring GPU libraries."""

import argparse
import json
import math
from pathlib import Path


def compare(base, head):
    for key in (
        "schema_version",
        "world_size",
        "gpu",
        "torch",
        "hip",
        "triton",
        "tritonblas",
        "dtype",
        "config",
        "warmup",
        "iterations",
        "trials",
    ):
        if base[key] != head[key]:
            raise ValueError(f"Cannot compare reports with different {key}.")
    if base["label"] != "base" or head["label"] != "head":
        raise ValueError("Expected base and head reports in that order.")

    def index(report):
        rows = {}
        for row in report["results"]:
            key = (row["m"], row["n"], row["k_local"], row["implementation"])
            if key in rows:
                raise ValueError(f"Duplicate benchmark case: {key}")
            if not row["validated"] or not math.isfinite(row["wall_ms"]) or row["wall_ms"] <= 0:
                raise ValueError(f"Invalid or unvalidated benchmark case: {key}")
            rows[key] = row
        return rows

    before, after = index(base), index(head)
    if not before or before.keys() != after.keys():
        raise ValueError("Reports must contain the same nonempty set of benchmark cases.")
    lines = [
        f"### {head['world_size']} ranks — {head['gpu']}",
        "",
        f"Base `{base['commit']}`; head `{head['commit']}`.",
        "",
        "| M × N × local K | Implementation | Base ms | Head ms | Base/head | Latency change |",
        "|---|---|---:|---:|---:|---:|",
    ]
    speedups = []
    for key in sorted(before):
        old, new = before[key], after[key]
        ratio = old["wall_ms"] / new["wall_ms"]
        change = (new["wall_ms"] / old["wall_ms"] - 1) * 100
        lines.append(
            f"| {key[0]} × {key[1]} × {key[2]} | {key[3]} | {old['wall_ms']:.4f} | "
            f"{new['wall_ms']:.4f} | {ratio:.3f}x | {change:+.1f}% |"
        )
        if key[3] != "torch_mm_rccl":
            speedups.append(ratio)
    if not speedups:
        raise ValueError("Reports must include at least one fused benchmark case.")
    geometric_mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    lines += [
        "",
        f"Geometric mean base/head across all {len(speedups)} fused cases: **{geometric_mean:.3f}x**.",
        "Ratios above 1 mean the PR is faster; positive latency changes mean slower.",
        "The unfused torch.mm + RCCL control is excluded from the geometric mean.",
        "",
        "Wall latency includes managed preparation, host coordination, and completion on all ranks.",
        "Raw JSON also records GPU stream elapsed time, which includes launch gaps and is not kernel-only timing.",
        "Medians summarize repeated trials, not statistical significance; rerun before publishing a performance claim.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sections = ["# matmul_all_reduce workspace reuse", ""]
    paths = sorted(args.results_dir.glob("base_*.json"))
    if not paths:
        raise ValueError("No base benchmark reports found.")
    for path in paths:
        head_path = path.with_name(path.name.replace("base_", "head_", 1))
        sections.append(compare(json.loads(path.read_text()), json.loads(head_path.read_text())))
    args.output.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
