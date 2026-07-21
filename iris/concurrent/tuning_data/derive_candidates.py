# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Derive the universally-good candidate set from the collected corpus.

Reads every ``data/<arch>_world<W>.json`` produced by ``collect_corpus.py`` and,
per architecture, builds an ORDERED list of ``(split_frac, gemm_block)`` configs
such that benchmarking the first ``k`` of them recovers the measured optimum on
as many corpus shapes as possible. The ordering is a greedy set-cover: repeatedly
pick the config that is "good enough" (within ``TOL`` of the per-shape best) on
the most so-far-uncovered shapes.

This is pure Python (no GPU/torch). Run after collecting data::

    python -m iris.concurrent.tuning_data.derive_candidates

Writes ``candidate_set.json`` and prints a coverage report (top-k vs %-optimal).
"""

import glob
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(__file__)
TOL = 0.03  # a config "covers" a shape if within 3% of that shape's measured best


def _canon(frac, block):
    return (round(float(frac), 3), tuple(block))


def load_rows_by_arch(data_dir):
    """arch -> list of shape rows (each row: best + grid across world sizes)."""
    by_arch = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path) as f:
            payload = json.load(f)
        if payload.get("quick"):
            continue
        arch = payload["arch"]
        cu = payload["cu_count"]
        for row in payload["results"]:
            if not row.get("best"):
                continue
            # config -> ms map for this shape, keyed by (frac, block)
            cfgms = {}
            for g in row["grid"]:
                if g["ms"] != g["ms"]:  # nan
                    continue
                cfgms[_canon(g["frac"], g["gemm_block"])] = g["ms"]
            if not cfgms:
                continue
            by_arch[arch].append(
                {
                    "world": payload["world_size"],
                    "cu": cu,
                    "key": f"{payload['world_size']}:{row['collective']}:{row['M']}x{row['N']}x{row['K']}:{row['comm_m']}x{row['comm_n']}",
                    "best_ms": row["best"]["ms"],
                    "cfgms": cfgms,
                }
            )
    return by_arch


def greedy_order(rows):
    """Greedy set-cover ordering of configs over shapes (within TOL of best)."""
    # candidate universe = all configs that appear
    universe = set()
    for r in rows:
        universe |= set(r["cfgms"].keys())
    # for each shape, the set of configs that are within TOL of its best
    covers = []
    for r in rows:
        thr = r["best_ms"] * (1.0 + TOL)
        good = {c for c, ms in r["cfgms"].items() if ms <= thr}
        covers.append(good)

    order = []
    uncovered = set(range(len(rows)))
    remaining = set(universe)
    while uncovered and remaining:
        # pick config covering the most uncovered shapes; tie-break by mean rel-slowdown
        best_cfg, best_gain, best_pen = None, -1, 1e9
        for c in remaining:
            gain = sum(1 for i in uncovered if c in covers[i])
            # penalty: mean rel cost across ALL shapes where present (prefer robust)
            rels = [rows[i]["cfgms"][c] / rows[i]["best_ms"] for i in range(len(rows)) if c in rows[i]["cfgms"]]
            pen = sum(rels) / len(rels) if rels else 1e9
            if gain > best_gain or (gain == best_gain and pen < best_pen):
                best_cfg, best_gain, best_pen = c, gain, pen
        if best_gain <= 0:
            break
        order.append(best_cfg)
        remaining.discard(best_cfg)
        uncovered = {i for i in uncovered if best_cfg not in covers[i]}
    # append any leftover uncovered shapes' individual best configs
    for i in sorted(uncovered):
        bc = min(rows[i]["cfgms"], key=lambda c: rows[i]["cfgms"][c])
        if bc not in order:
            order.append(bc)
    return order


def coverage_report(rows, order):
    """For k=1..len(order): fraction of shapes within TOL, and mean/max slowdown."""
    report = []
    for k in range(1, len(order) + 1):
        topk = order[:k]
        within = 0
        slow = []
        for r in rows:
            present = [r["cfgms"][c] for c in topk if c in r["cfgms"]]
            if not present:
                continue
            best_of_k = min(present)
            slow.append(best_of_k / r["best_ms"])
            if best_of_k <= r["best_ms"] * (1.0 + TOL):
                within += 1
        report.append(
            {
                "k": k,
                "pct_within_tol": round(100.0 * within / len(rows), 1),
                "mean_slowdown": round(sum(slow) / len(slow), 4),
                "max_slowdown": round(max(slow), 4),
            }
        )
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser()
    # Raw corpus lives in origami_comms, not in this repo; default to a local
    # ./data/ if present (e.g. a scratch regen), else require --data.
    ap.add_argument(
        "--data",
        default=os.path.join(_HERE, "data"),
        help="directory of <arch>_world<W>.json corpus files (origami_comms)",
    )
    args = ap.parse_args()

    data_dir = args.data
    by_arch = load_rows_by_arch(data_dir)
    if not by_arch:
        print(f"no corpus data found in {data_dir}; point --data at the origami_comms corpus dir")
        return

    out = {"_tol": TOL, "_note": "ordered greedy set-cover of (split_frac, gemm_block); benchmark top-k."}
    for arch, rows in sorted(by_arch.items()):
        order = greedy_order(rows)
        rep = coverage_report(rows, order)
        out[arch] = {
            "n_shapes": len(rows),
            "worlds": sorted({r["world"] for r in rows}),
            "order": [{"split_frac": c[0], "gemm_block": list(c[1])} for c in order],
            "coverage": rep,
        }
        print(f"\n=== {arch}  ({len(rows)} shape-points across worlds {sorted({r['world'] for r in rows})}) ===")
        for c, rp in zip(order, rep):
            print(
                f"  k={rp['k']:>2}  frac={c[0]:<5} blk={tuple(c[1])!s:<16} -> {rp['pct_within_tol']:>5}% within {int(TOL * 100)}%  mean_slow={rp['mean_slowdown']}  max_slow={rp['max_slowdown']}"
            )

    out_path = os.path.join(_HERE, "candidate_set.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
