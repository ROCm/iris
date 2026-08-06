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
    p.add_argument("json", nargs="+", help="one roofline json per world size")
    p.add_argument("-o", "--out", default="roofline")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_ws = {}
    for path in a.json:
        rows = json.load(open(path))
        w = rows[0]["world_size"]
        d = defaultdict(dict)
        for r in rows:
            d[r["algo"]][r["M"]] = r
        per_ws[w] = d
    ws = max(per_ws)
    by = per_ws[ws]
    byte_ratio = ws / (2 * (ws - 1) / ws)

    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black",
        "savefig.facecolor": "black", "text.color": "white",
        "axes.labelcolor": "white", "xtick.color": "white",
        "ytick.color": "white", "axes.edgecolor": "#555555",
        "grid.color": "#333333", "font.size": 10,
    })

    fig, axes = plt.subplots(2, 3, figsize=(21, 9))

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

    # ---- 4. the crossover rule, across world sizes ----
    # byte_ratio(ws) = ws / (2(ws-1)/ws) = ws^2 / (2(ws-1)):
    #   ws=2 -> 2.00, ws=4 -> 2.67, ws=8 -> 4.57
    # so the rule predicts the crossover MOVES with ws. Overlaying the
    # measured efficiency ratio per ws against its own byte ratio tests that.
    ax = axes[1][1]
    shades = {2: "#fcffa4", 4: "#f98e09", 8: "#bc3754"}
    crossings = {}
    for w in sorted(per_ws):
        b = per_ws[w]
        if "RCCL all_reduce" not in b or "iris one-shot" not in b:
            continue
        br = w / (2 * (w - 1) / w)
        Ms = sorted(set(b["RCCL all_reduce"]) & set(b["iris one-shot"]))
        eff = [b["iris one-shot"][m]["gbs"] / b["RCCL all_reduce"][m]["gbs"]
               for m in Ms]
        c = shades.get(w, "#57106e")
        ax.plot(Ms, eff, color=c, marker="o", lw=2,
                label=f"ws={w}: efficiency ratio")
        ax.axhline(br, color=c, ls="--", lw=1.4, alpha=0.8)
        ax.text(Ms[0], br * 1.04, f"byte ratio {br:.2f}", color=c, fontsize=7)
        for i in range(1, len(Ms)):
            if (eff[i - 1] - br) * (eff[i] - br) < 0:
                x0, x1 = np.log2(Ms[i - 1]), np.log2(Ms[i])
                cx = 2 ** (x0 + (br - eff[i - 1]) * (x1 - x0) /
                           (eff[i] - eff[i - 1]))
                crossings[w] = cx
                ax.axvline(cx, color=c, ls=":", lw=1.2, alpha=0.8)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("ratio  (solid = measured, dashed = predicted threshold)")
    ax.grid(alpha=0.3)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper right")

    # ---- 5. fused two-shot across every world size where it is VALID ----
    # Its valid region widens as ws shrinks: two-shot needs M/ws to be a
    # tileable shard, so the smallest runnable M is 128 at ws=2, 256 at ws=4,
    # 512 at ws=8. Plotting only ws=8 hides most of where it can run.
    ax = axes[0][2]
    ax.axhline(LINE_GBS, color="white", ls=":", lw=1.2)
    for w in sorted(per_ws):
        b = per_ws[w].get("fused two-shot (comm)", {})
        if not b:
            continue
        Ms = sorted(b)
        ax.plot(Ms, [b[m]["gbs"] for m in Ms], color=shades.get(w, "#57106e"),
                marker="^", lw=2, label=f"fused two-shot comm, ws={w}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("fused comm GB/s")
    ax.set_ylim(0, LINE_GBS * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper left")

    # ---- 6. which algorithm actually wins, over the whole (ws, M) grid ----
    ax = axes[1][2]
    cols = {"torch": "#000004", "one-shot": "#bc3754", "fused": "#f98e09"}
    wss = sorted(per_ws)
    allM = sorted({m for w in wss for m in per_ws[w]["torch mm+AR (E2E)"]})
    for yi, w in enumerate(wss):
        b = per_ws[w]
        for xi, m in enumerate(allM):
            cand = []
            for key, nm in (("torch mm+AR (E2E)", "torch"),
                            ("two-kernel one-shot (E2E)", "one-shot"),
                            ("fused two-shot (E2E)", "fused")):
                if m in b.get(key, {}):
                    cand.append((b[key][m]["ms"], nm))
            if not cand:
                continue
            ms, nm = min(cand)
            ax.add_patch(plt.Rectangle((xi - .5, yi - .5), 1, 1,
                                       color=cols[nm], ec="#222222"))
            base = b["torch mm+AR (E2E)"][m]["ms"]
            ax.text(xi, yi, f"{base/ms:.2f}", ha="center", va="center",
                    color="white" if nm != "torch" else "#888888", fontsize=7)
    ax.set_xlim(-.5, len(allM) - .5); ax.set_ylim(-.5, len(wss) - .5)
    ax.set_xticks(range(len(allM))); ax.set_xticklabels(allM, fontsize=8)
    ax.set_yticks(range(len(wss))); ax.set_yticklabels([f"ws={w}" for w in wss])
    ax.set_xlabel("M (tokens)")
    ax.set_ylabel("world size")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k)
                       for k, c in cols.items()],
              facecolor="black", edgecolor="#555555", labelcolor="white",
              fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{a.out}_roofline.png", dpi=140)
    print(f"wrote {a.out}_roofline.png")
    for w in sorted(crossings):
        print(f"  ws={w}: byte_ratio={w/(2*(w-1)/w):.2f}  "
              f"crossover M ~= {crossings[w]:.0f}")
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
