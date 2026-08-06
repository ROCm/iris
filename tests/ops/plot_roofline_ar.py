#!/usr/bin/env python3
"""Roofline plots for GEMM+AllReduce, and the rule that predicts every crossover.

Time = bytes / bandwidth. The algorithms differ in BOTH terms and in opposite
directions, which is why comparing either one alone has been misleading all
study:

  one-shot   moves ws/(2(ws-1)/ws) = 4.57x more bytes than RCCL at ws=8,
             but sustains 63-85% of XGMI line rate
  RCCL       moves the minimum bytes, but sustains only 8-58%
  two-shot   moves RCCL's bytes at 15-29% -- worst of both

So one-shot wins exactly while

    efficiency_ratio = GBs(one-shot) / GBs(RCCL)  >  byte_ratio = 4.57

That single inequality predicts the measured crossover between M=256 and
M=512 without any fitting, and it is what the fourth panel plots.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

LINE_GBS = 448.0

STYLE = {
    "RCCL all_reduce":           ("#f98e09", "-",  "o"),
    "iris one-shot":             ("#bc3754", "-",  "s"),
    "fused two-shot (comm)":     ("#57106e", "-",  "^"),
    "torch mm+AR (E2E)":         ("#f98e09", "--", "o"),
    "two-kernel one-shot (E2E)": ("#bc3754", "--", "s"),
    "fused two-shot (E2E)":      ("#57106e", "--", "^"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("-o", "--out", default="roofline")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(a.json))
    ws = rows[0]["world_size"]
    byte_ratio = ws / (2 * (ws - 1) / ws)

    by = defaultdict(dict)
    for r in rows:
        by[r["algo"]][r["M"]] = r

    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black",
        "savefig.facecolor": "black", "text.color": "white",
        "axes.labelcolor": "white", "xtick.color": "white",
        "ytick.color": "white", "axes.edgecolor": "#555555",
        "grid.color": "#333333", "font.size": 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # ---- 1. achieved bandwidth vs line rate ----
    ax = axes[0][0]
    ax.axhline(LINE_GBS, color="white", ls=":", lw=1.4)
    ax.text(40, LINE_GBS * 0.93, f"XGMI line rate {LINE_GBS:.0f} GB/s",
            color="white", fontsize=8)
    for algo in ("RCCL all_reduce", "iris one-shot", "fused two-shot (comm)"):
        if algo not in by:
            continue
        Ms = sorted(by[algo])
        c, ls, mk = STYLE[algo]
        ax.plot(Ms, [by[algo][m]["gbs"] for m in Ms], color=c, ls=ls,
                marker=mk, lw=2, label=algo)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("achieved GB/s on algorithmic bytes")
    ax.set_ylim(0, LINE_GBS * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper left")

    # ---- 2. bytes actually moved ----
    ax = axes[0][1]
    for algo in ("RCCL all_reduce", "iris one-shot"):
        Ms = sorted(by[algo])
        c, ls, mk = STYLE[algo]
        lbl = ("minimum bytes 2(ws-1)/ws*MN" if "RCCL" in algo
               else f"one-shot ws*MN ({byte_ratio:.2f}x more)")
        ax.plot(Ms, [by[algo][m]["moved_mb"] for m in Ms], color=c, ls=ls,
                marker=mk, lw=2, label=lbl)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("MB moved per rank")
    ax.grid(alpha=0.3)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper left")

    # ---- 3. end-to-end time ----
    ax = axes[1][0]
    for algo in ("torch mm+AR (E2E)", "two-kernel one-shot (E2E)",
                 "fused two-shot (E2E)"):
        if algo not in by:
            continue
        Ms = sorted(by[algo])
        c, ls, mk = STYLE[algo]
        ax.plot(Ms, [by[algo][m]["ms"] for m in Ms], color=c, ls=ls,
                marker=mk, lw=2, label=algo)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("end-to-end ms")
    ax.grid(alpha=0.3)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper left")

    # ---- 4. the crossover rule ----
    ax = axes[1][1]
    Ms = sorted(set(by["RCCL all_reduce"]) & set(by["iris one-shot"]))
    eff = [by["iris one-shot"][m]["gbs"] / by["RCCL all_reduce"][m]["gbs"]
           for m in Ms]
    ax.plot(Ms, eff, color="#bc3754", marker="s", lw=2.2,
            label="efficiency ratio  one-shot / RCCL")
    ax.axhline(byte_ratio, color="#f98e09", ls="--", lw=2,
               label=f"byte ratio  {byte_ratio:.2f}x")
    ax.fill_between(Ms, byte_ratio, eff,
                    where=[e >= byte_ratio for e in eff],
                    color="#bc3754", alpha=0.25)
    ax.fill_between(Ms, byte_ratio, eff,
                    where=[e < byte_ratio for e in eff],
                    color="#f98e09", alpha=0.25)
    # where the two curves cross, in log-M
    cross = None
    for i in range(1, len(Ms)):
        if (eff[i - 1] - byte_ratio) * (eff[i] - byte_ratio) < 0:
            x0, x1 = np.log2(Ms[i - 1]), np.log2(Ms[i])
            y0, y1 = eff[i - 1], eff[i]
            cross = 2 ** (x0 + (byte_ratio - y0) * (x1 - x0) / (y1 - y0))
    if cross:
        ax.axvline(cross, color="white", ls=":", lw=1.4)
        ax.text(cross * 1.05, max(eff) * 0.85, f"crossover M≈{cross:.0f}",
                color="white", fontsize=9)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("ratio")
    ax.grid(alpha=0.3)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{a.out}_roofline.png", dpi=140)
    print(f"wrote {a.out}_roofline.png")
    if cross:
        print(f"predicted crossover: M ~= {cross:.0f}")
    for m in Ms:
        r_, o_ = by["RCCL all_reduce"][m], by["iris one-shot"][m]
        f_ = by.get("fused two-shot (comm)", {}).get(m)
        print(f"  M={m:5d}  RCCL {r_['gbs']:6.1f} GB/s ({r_['pct_line']:4.0f}%)"
              f"   one-shot {o_['gbs']:6.1f} ({o_['pct_line']:4.0f}%)"
              f"   ratio {o_['gbs']/r_['gbs']:5.2f}"
              + (f"   fused-comm {f_['gbs']:6.1f} ({f_['pct_line']:3.0f}%)"
                 if f_ else "   fused-comm    n/a"))


if __name__ == "__main__":
    main()
