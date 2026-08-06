#!/usr/bin/env python3
"""Plot the fused GEMM+AR trace so the overlap is visible instead of inferred.

Summary statistics said "GEMM finishes at 92us of a 277us kernel, so 75% of
the machine is idle" -- and acting on that made the kernel 1.8x slower. The
statistics were not wrong, the story built on them was. These plots show what
each work-group is actually doing over time so the story has to match the
picture.

Four panels:
  1. Gantt      -- one row per work-group, one bar per tile, coloured by pool.
                   Shows directly which WGs are busy when and where the gaps are.
  2. Occupancy  -- work-groups active per pool over time. The area under each
                   curve is machine utilisation; gaps between curves are the
                   pipeline not filling.
  3. Spin/work  -- per pool, time spent waiting on a flag vs moving data.
  4. Tile flow  -- per tile: GEMM done, RS done, AG done. The vertical distance
                   between the curves is pipeline latency per tile; the slope
                   is throughput.

Run on the node with --dump to write the raw arrays, then plot anywhere.
"""

import argparse
import os

import numpy as np


def plot(npz_path, out_prefix, world_size, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = np.load(npz_path)
    freq = float(d["freq_mhz"])
    MAXV = np.iinfo(np.int64).max

    def clean(a):
        a = a.astype(np.int64)
        return np.where((a == MAXV) | (a <= 0), -1, a)

    g0, g1 = clean(d["gemm_beg"]), clean(d["gemm_end"])
    r0, rr, r1 = clean(d["rs_beg"]), clean(d["rs_ready"]), clean(d["rs_end"])
    a0, ar, a1 = clean(d["ag_beg"]), clean(d["ag_ready"]), clean(d["ag_end"])
    gwg, rwg, awg = d["gemm_wg"].astype(int), d["rs_wg"].astype(int), d["ag_wg"].astype(int)

    valid = np.concatenate([g0[g0 > 0], r0[r0 > 0], a0[a0 > 0]])
    t0 = valid.min()
    us = lambda x: np.where(x > 0, (x - t0) / freq, np.nan)

    G0, G1 = us(g0), us(g1)
    R0, RR, R1 = us(r0), us(rr), us(r1)
    A0, AR, A1 = us(a0), us(ar), us(a1)
    tmax = np.nanmax(np.concatenate([G1, R1, A1]))

    C = {"GEMM": "#f98e09", "RS": "#bc3754", "AG": "#57106e", "spin": "#4a4a4a"}
    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black",
        "savefig.facecolor": "black", "text.color": "white",
        "axes.labelcolor": "white", "xtick.color": "white",
        "ytick.color": "white", "axes.edgecolor": "#555555",
        "font.size": 9,
    })

    # ---------- 1. Gantt ----------
    fig, ax = plt.subplots(figsize=(13, 7))
    for wg, b, e, name in ((gwg, G0, G1, "GEMM"), (rwg, R0, R1, "RS"),
                           (awg, A0, A1, "AG")):
        m = ~np.isnan(b) & ~np.isnan(e)
        ax.barh(wg[m], (e - b)[m], left=b[m], height=0.9,
                color=C[name], linewidth=0)
    # the spin portion, drawn over the bar so waiting is visually distinct
    for wg, b, ready in ((rwg, R0, RR), (awg, A0, AR)):
        m = ~np.isnan(b) & ~np.isnan(ready)
        ax.barh(wg[m], (ready - b)[m], left=b[m], height=0.9,
                color=C["spin"], linewidth=0)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("work-group id")
    ax.set_xlim(0, tmax)
    ax.legend(handles=[Patch(color=C[k], label=v) for k, v in
                       (("GEMM", "GEMM"), ("RS", "reduce-scatter"),
                        ("AG", "all-gather"), ("spin", "spinning on flag"))],
              facecolor="black", edgecolor="#555555", labelcolor="white",
              loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_gantt.png", dpi=140)
    plt.close(fig)

    # ---------- 2. Occupancy ----------
    grid = np.linspace(0, tmax, 900)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for b, e, name in ((G0, G1, "GEMM"), (RR, R1, "RS"), (AR, A1, "AG")):
        m = ~np.isnan(b) & ~np.isnan(e)
        occ = ((grid[:, None] >= b[m]) & (grid[:, None] < e[m])).sum(axis=1)
        ax.fill_between(grid, occ, color=C[name], alpha=0.55, label=name)
        ax.plot(grid, occ, color=C[name], lw=1.2)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("tiles in flight")
    ax.set_xlim(0, tmax)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_occupancy.png", dpi=140)
    plt.close(fig)

    # ---------- 3. Spin vs work ----------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names, spins, works = [], [], []
    for b, ready, e, name in ((R0, RR, R1, "RS"), (A0, AR, A1, "AG")):
        m = ~np.isnan(b) & ~np.isnan(ready) & ~np.isnan(e)
        names.append(name)
        spins.append(np.nansum(ready[m] - b[m]) / 1e3)
        works.append(np.nansum(e[m] - ready[m]) / 1e3)
    x = np.arange(len(names))
    ax.bar(x, spins, color=C["spin"], label="spin (waiting on producer)")
    ax.bar(x, works, bottom=spins, color=C["RS"], label="work (moving data)")
    for i, (s_, w_) in enumerate(zip(spins, works)):
        ax.text(i, s_ + w_, f"  {100*s_/(s_+w_):.0f}% spin",
                ha="center", va="bottom", color="white")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("aggregate WG-time (ms)")
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_spin.png", dpi=140)
    plt.close(fig)

    # ---------- 4. Tile flow ----------
    fig, ax = plt.subplots(figsize=(13, 4.2))
    order = np.argsort(np.where(np.isnan(G1), 1e18, G1))
    idx = np.arange(len(order))
    for arr, name in ((G1, "GEMM"), (R1, "RS"), (A1, "AG")):
        ax.plot(np.sort(arr[order][~np.isnan(arr[order])]),
                idx[:np.sum(~np.isnan(arr[order]))],
                color=C[name], lw=1.8, label=f"{name} complete")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("tiles completed")
    ax.set_xlim(0, tmax)
    ax.legend(facecolor="black", edgecolor="#555555", labelcolor="white")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_flow.png", dpi=140)
    plt.close(fig)

    print(f"wrote {out_prefix}_{{gantt,occupancy,spin,flow}}.png")

    # numbers the plots should be read against
    gw = np.unique(gwg[~np.isnan(G0)])
    rw = np.unique(rwg[~np.isnan(R0)])
    aw = np.unique(awg[~np.isnan(A0)])
    print(f"  work-groups seen: GEMM {len(gw)}  RS {len(rw)}  AG {len(aw)}")
    print(f"  GEMM  {np.nanmin(G0):7.1f} -> {np.nanmax(G1):7.1f} us")
    print(f"  RS    {np.nanmin(R0):7.1f} -> {np.nanmax(R1):7.1f} us")
    print(f"  AG    {np.nanmin(A0):7.1f} -> {np.nanmax(A1):7.1f} us")
    print(f"  wall  {tmax:.1f} us")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("-o", "--out", default="ar_trace")
    p.add_argument("-r", "--world_size", type=int, default=8)
    p.add_argument("-m", type=int, default=2048)
    a = p.parse_args()
    plot(a.npz, a.out, a.world_size, a.m)


if __name__ == "__main__":
    main()
