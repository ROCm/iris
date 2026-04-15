#!/usr/bin/env python3
"""
Analyze and plot cross-warp coalescing from address dump.

Reads coalescing_results.json (produced by coalescing_probe.py) and generates:
  1. Scatter plot: X = flat thread index, Y = byte offset from base, colored by warp
  2. Cache-line heatmap: which 128B lines each warp touches
  3. Summary table printed to stdout

Usage:
    python3 analyze_coalescing.py [--input coalescing_results.json] [--output coalescing_plot.png]
"""
import argparse
import json
import sys

import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def print_summary(results):
    cfg = results["config"]
    print("=" * 60)
    print("Cross-warp coalescing analysis")
    print("=" * 60)
    print(f"  BLOCK_SIZE_M      = {cfg['block_size_m']}")
    print(f"  BLOCK_SIZE_N      = {cfg['block_size_n']}")
    print(f"  THREADS_PER_WARP  = {cfg['threads_per_warp']}")
    print(f"  WARPS_PER_CTA     = {cfg['warps_per_cta']}")
    print(f"  ELEMS_PER_THREAD  = {cfg['elems_per_thread']}")
    print(f"  dtype             = {cfg['dtype']} ({cfg['elem_bytes']}B/elem)")
    print(f"  TOTAL_ELEMS       = {cfg['total_elems']}")
    print()

    # Per-warp table
    print(f"{'Warp':>6s}  {'Byte range':>20s}  {'Cache lines (128B)':>20s}  {'# lines':>8s}")
    print("-" * 60)
    for w in results["per_warp"]:
        br = w["byte_range"]
        cl = w["cache_lines_128B"]
        print(f"{w['warp_id']:>6d}  [{br[0]:>8d}, {br[1]:>8d})  [{cl[0]:>8d}, {cl[1]:>8d}]  {w['num_cache_lines']:>8d}")

    print()

    # Cross-warp adjacency
    per_warp = results["per_warp"]
    for i in range(len(per_warp) - 1):
        this_last = per_warp[i]["cache_lines_128B"][1]
        next_first = per_warp[i + 1]["cache_lines_128B"][0]
        adj = next_first == this_last + 1
        sym = "OK" if adj else "GAP"
        print(f"  Warp {i} -> {i+1}: line {this_last} -> {next_first}  [{sym}]")

    print()
    adj = results.get("cross_warp_adjacent", False)
    mono = results.get("monotonically_increasing", False)
    print(f"  Cross-warp adjacent:          {'YES' if adj else 'NO'}")
    print(f"  Monotonically increasing:     {'YES' if mono else 'NO'}")

    if adj and mono:
        print("\n  VERDICT: Hardware can coalesce adjacent 128B lines into 256B transactions.")
    elif adj:
        print("\n  VERDICT: Adjacent but not monotonic — coalescing possible, ordering unusual.")
    else:
        print("\n  VERDICT: Gaps between warp ranges — 256B coalescing may not occur at boundaries.")
    print()


def plot_scatter(results, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available — skipping plot generation.", file=sys.stderr)
        print("Install with: pip install matplotlib", file=sys.stderr)
        return False

    cfg = results["config"]
    addrs = np.array(results["addrs_flat"], dtype=np.int64)
    base = results["base_addr"]
    offsets = addrs - base  # byte offsets

    num_warps = cfg["warps_per_cta"]
    tpw = cfg["threads_per_warp"]
    ept = cfg["elems_per_thread"]
    total = cfg["total_elems"]

    # Assign warp ID to each element
    # BlockedLayout([EPT], [TPW], [WPC], [0]):
    #   flat_idx = warp * (TPW * EPT) + thread * EPT + elem
    warp_ids = np.zeros(total, dtype=int)
    for w in range(num_warps):
        start = w * tpw * ept
        end = start + tpw * ept
        warp_ids[start:end] = w

    flat_indices = np.arange(total)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_warps, 10)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # --- Top: scatter plot ---
    ax = axes[0]
    for w in range(num_warps):
        mask = warp_ids == w
        ax.scatter(
            flat_indices[mask], offsets[mask],
            c=[colors[w]], s=4, alpha=0.7, label=f"Warp {w}",
            edgecolors="none",
        )

    # Mark 128B cache line boundaries
    max_offset = offsets.max() + cfg["elem_bytes"]
    for cl_byte in range(0, int(max_offset) + 128, 128):
        ax.axhline(cl_byte, color="gray", linewidth=0.3, alpha=0.4)

    # Mark 256B super-line boundaries
    for sl_byte in range(0, int(max_offset) + 256, 256):
        ax.axhline(sl_byte, color="red", linewidth=0.5, alpha=0.3, linestyle="--")

    ax.set_xlabel("Flat element index (BlockedLayout order)")
    ax.set_ylabel("Byte offset from base address")
    ax.set_title(
        f"Per-thread store addresses — "
        f"BLOCK_SIZE_M={cfg['block_size_m']}, BLOCK_SIZE_N={cfg['block_size_n']}, "
        f"{cfg['dtype']}, {num_warps} warps x {tpw} threads"
    )
    ax.legend(loc="upper left", markerscale=3)

    # Add custom legend entries for grid lines
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=0.5, label="128B cache line"),
        Line2D([0], [0], color="red", linewidth=0.5, linestyle="--", label="256B super-line"),
    ]
    ax.add_artist(ax.legend(handles=legend_elements, loc="lower right", fontsize=8))

    # --- Bottom: cache line heatmap ---
    ax2 = axes[1]
    max_cl = int(max_offset) // 128 + 1
    heatmap = np.zeros((num_warps, max_cl), dtype=int)
    for i, (off, w) in enumerate(zip(offsets, warp_ids)):
        cl = int(off) // 128
        heatmap[w, cl] += 1

    im = ax2.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax2.set_xlabel("128B cache line index")
    ax2.set_ylabel("Warp ID")
    ax2.set_title("Elements per 128B cache line per warp")
    ax2.set_yticks(range(num_warps))
    fig.colorbar(im, ax=ax2, label="# elements", shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-warp coalescing")
    parser.add_argument("--input", default="coalescing_results.json", help="Input JSON from coalescing_probe.py")
    parser.add_argument("--output", default="coalescing_plot.png", help="Output plot image")
    args = parser.parse_args()

    results = load_results(args.input)
    print_summary(results)
    plot_scatter(results, args.output)


if __name__ == "__main__":
    main()
