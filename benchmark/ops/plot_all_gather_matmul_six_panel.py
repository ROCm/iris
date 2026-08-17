#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Generate the six-panel all-gather-matmul paper figure and CSV/TikZ sources."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


BASELINE = ("tritonblas_rcclbaseline", "TritonBLAS + RCCL", "#4C78A8")
BENCHMARKS = (
    ("hbm_buffer", "Fused kernel", "#F58518"),
    ("copy_engine_host", "GPU-triggered SDMA", "#54A24B"),
    ("copy_engine_device", "GPU-enqueued SDMA", "#B279A2"),
)


@dataclass(frozen=True)
class Panel:
    model: str
    operation: str
    title: str
    csv_name: str | None


PANELS = (
    Panel("gpt_oss_120b", "attn_out", "GPT-OSS 120B: Attention", "gpt_oss_120b_attn_out.csv"),
    Panel("deepseek_v4", "attn_out", "DeepSeek V4: Attention", "deepseek_v4_attn_out.csv"),
    Panel("llama3_70b", "attn_out", "Llama 3 70B: Attention", "llama3_70b_attn_out.csv"),
    Panel("gpt_oss_120b", "expert_mlp_down", "GPT-OSS 120B: Expert MLP", None),
    Panel(
        "deepseek_v4",
        "expert_mlp_down",
        "DeepSeek V4: Expert MLP",
        "deepseek_v4_expert_mlp_down.csv",
    ),
    Panel("llama3_70b", "mlp_down", "Llama 3 70B: Dense MLP", "llama3_70b_mlp_down.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the six-panel all-gather-matmul paper figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Merged all-gather-matmul JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_all_gather_six_panel"))
    parser.add_argument("--m-values", type=int, nargs="+", default=(512, 2048, 8192))
    parser.add_argument("--absolute", action="store_true", help="Plot TFLOP/s rather than speedup vs TritonBLAS")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _label(panel: Panel, m: int) -> str:
    suffix = {2048: "2k", 8192: "8k", 32768: "32k"}.get(m, str(m))
    return f"{panel.model}_{panel.operation}_{suffix}"


def _load_rows(results: list[dict], panel: Panel, m_values: list[int], normalized: bool) -> list[dict]:
    by_label = {row.get("label"): row for row in results}
    rows = []
    for m in m_values:
        label = _label(panel, m)
        if label not in by_label:
            raise ValueError(f"Missing result {label!r}")
        data = by_label[label].get("benchmarks") or {}
        baseline_data = data.get(BASELINE[0]) or {}
        baseline = baseline_data.get("tflops")
        if not isinstance(baseline, (int, float)) or baseline <= 0:
            raise ValueError(f"Missing positive tflops for {label}.{BASELINE[0]}")
        row = {"M": m, BASELINE[0]: 1.0 if normalized else baseline}
        for key, _, _ in BENCHMARKS:
            value = (data.get(key) or {}).get("tflops")
            if not isinstance(value, (int, float)):
                if key == "hbm_buffer":
                    row[key] = float("nan")
                    continue
                raise ValueError(f"Missing numeric tflops for {label}.{key}")
            row[key] = value / baseline if normalized else value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["M", BASELINE[0], *(key for key, _, _ in BENCHMARKS)]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, panel_rows: list[tuple[Panel, list[dict] | None]], m_values: list[int], normalized: bool):
    ylabel = "Speedup vs. TritonBLAS" if normalized else "TFLOP/s"
    lines = [
        r"\documentclass[tikz,border=3pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{hbmbufferorange}{HTML}{F58518}",
        r"\definecolor{triggergreen}{HTML}{54A24B}",
        r"\definecolor{enqueuepurple}{HTML}{B279A2}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 2, horizontal sep=0.75cm, vertical sep=1.05cm},",
        r"  width=0.31\textwidth, height=0.27\textwidth, ybar, ymin=0,",
        "  symbolic x coords={" + ",".join(str(m) for m in m_values) + "},",
        r"  xtick=data, enlarge x limits=0.22, unbounded coords=discard,",
        r"  grid=major, major grid style={gray!25}, axis on top,",
        r"  tick label style={font=\small}, label style={font=\small}, title style={font=\small},",
        r"  legend style={font=\small, draw=none, legend columns=3},",
        r"]",
    ]
    colors = ("hbmbufferorange", "triggergreen", "enqueuepurple")
    for index, (panel, rows) in enumerate(panel_rows):
        options = [f"title={{{panel.title}}}"]
        if index % 3 == 0:
            options.append(f"ylabel={{{ylabel}}}")
        if index >= 3:
            options.append(r"xlabel={Batch Size}")
        if index == 0:
            options.append("legend to name=figurelegend")
        if rows is None:
            options.extend(("axis lines=none", "xtick=\\empty", "ytick=\\empty", "xmin=512", "xmax=8192", "ymax=1"))
        lines.append(r"\nextgroupplot[" + ", ".join(options) + "]")
        if rows is None:
            lines.append(r"\node[align=center,text=gray] at (axis description cs:0.5,0.5) {Unavailable\\(unsupported K tile)};")
            continue
        if normalized:
            coords = " ".join(f"({m},1)" for m in m_values)
            lines.append(rf"\addplot+[black!60,dashed,thick,mark=none,sharp plot,forget plot] coordinates {{{coords}}};")
        for bench_index, (key, name, _) in enumerate(BENCHMARKS):
            color = colors[bench_index]
            label_options = ""
            if normalized:
                label_options = (
                    r",point meta=y,nodes near coords={\pgfmathprintnumber[fixed,precision=2]{\pgfplotspointmeta}x}"
                    r",every node near coord/.append style={font=\tiny,anchor=south,yshift=1pt}"
                )
            lines.append(
                rf"\addplot+[fill={color},draw={color},bar width=7pt{label_options}, area legend] "
                rf"table[x=M,y={key},col sep=comma] {{data/{panel.csv_name}}};"
            )
            if index == 0:
                lines.append(rf"\addlegendentry{{{name}}}")
    lines.extend(
        (
            r"\end{groupplot}",
            r"\node[anchor=north] at ([yshift=-0.55cm]group c2r2.south) {\ref{figurelegend}};",
            r"\end{tikzpicture}",
            r"\end{document}",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def _plot(output_dir: Path, panel_rows, m_values: list[int], normalized: bool, dpi: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Run with benchmark/ops/env/bin/python to provide Matplotlib") from error
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9})
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    positions = list(range(len(m_values)))
    width = 0.8 / len(BENCHMARKS)
    handles = labels = None
    for axis, (panel, rows) in zip(axes.flat, panel_rows):
        axis.set_title(panel.title)
        if rows is None:
            axis.axis("off")
            axis.text(0.5, 0.5, "Unavailable\n(unsupported K tile)", ha="center", va="center", color="0.45")
            continue
        for index, (key, name, color) in enumerate(BENCHMARKS):
            offset = (index - (len(BENCHMARKS) - 1) / 2) * width
            values = [row[key] for row in rows]
            bars = axis.bar([x + offset for x in positions], values, width, label=name, color=color)
            if normalized:
                bar_labels = [f"{value:.2f}x" if value == value else "" for value in values]
                axis.bar_label(bars, labels=bar_labels, padding=2, fontsize=7)
        if normalized:
            axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1.5, zorder=1)
            axis.margins(y=0.18)
        axis.set_xticks(positions, [str(m) for m in m_values])
        axis.set_axisbelow(True)
        axis.grid(True, axis="y", alpha=0.25)
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()
    ylabel = "Speedup vs. TritonBLAS + RCCL" if normalized else "TFLOP/s"
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    for axis in axes[1, 1:]:
        axis.set_xlabel("Batch Size")
    figure.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    figure.savefig(output_dir / "all_gather_matmul_six_panel.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "all_gather_matmul_six_panel.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if len(set(args.m_values)) != len(args.m_values) or any(m <= 0 for m in args.m_values):
        raise SystemExit("error: --m-values must contain unique positive integers")
    try:
        with args.input.open() as file:
            results = json.load(file)
        if not isinstance(results, list):
            raise ValueError("Expected a top-level JSON list")
        normalized = not args.absolute
        panel_rows = [
            (panel, None if panel.csv_name is None else _load_rows(results, panel, args.m_values, normalized))
            for panel in PANELS
        ]
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for panel, rows in panel_rows:
        if rows is not None:
            _write_csv(data_dir / panel.csv_name, rows)
    _write_tex(args.output_dir / "all_gather_matmul_six_panel.tex", panel_rows, args.m_values, normalized)
    try:
        _plot(args.output_dir, panel_rows, args.m_values, normalized, args.dpi)
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Wrote all-gather-matmul figure, TeX source, and CSV data to {args.output_dir}")


if __name__ == "__main__":
    main()
