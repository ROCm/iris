#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Generate the six-panel matmul-all-reduce paper figure and its source data.

The script writes PDF/PNG plots, one CSV per panel, and a standalone PGFPlots
document that reads the same CSV files.

Example:
    benchmark/ops/env/bin/python benchmark/ops/plot_matmul_all_reduce_six_panel.py \
        benchmark/ops/model_sweep_results_matmul_all_reduce-merged.json \
        --output-dir benchmark/ops/paper_six_panel
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


BENCHMARKS = (
    ("tritonblas_rcclbaseline", "TritonBLAS + RCCL", "#4C78A8"),
    ("two_shot", "Fused two-shot", "#F58518"),
    ("copy_engine_device_two_shot", "GPU-triggered SDMA", "#54A24B"),
)
GPU_ENQUEUED_BENCHMARK = ("copy_engine_device_two_shot_gpu_init", "GPU-enqueued SDMA", "#B279A2")


@dataclass(frozen=True)
class Panel:
    model: str
    operation: str
    title: str
    csv_name: str


PANELS = (
    Panel("gpt_oss_120b", "attn_out", "GPT-OSS 120B: Attention", "gpt_oss_120b_attn_out.csv"),
    Panel("deepseek_v4", "attn_out", "DeepSeek V4: Attention", "deepseek_v4_attn_out.csv"),
    Panel("llama3_70b", "attn_out", "Llama 3 70B: Attention", "llama3_70b_attn_out.csv"),
    Panel(
        "gpt_oss_120b",
        "expert_mlp_down",
        "GPT-OSS 120B: Expert MLP",
        "gpt_oss_120b_expert_mlp_down.csv",
    ),
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
        description="Generate the six-panel matmul-all-reduce paper figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Merged model-sweep JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_six_panel"), help="Output directory")
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=(512, 2048, 8192),
        help="M values to include, in display order",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    parser.add_argument(
        "--normalize-to-tritonblas",
        action="store_true",
        help="Plot speedup relative to TritonBLAS + RCCL instead of absolute TFLOP/s",
    )
    parser.add_argument(
        "--include-gpu-enqueued",
        action="store_true",
        help="Include the GPU-enqueued SDMA copy-engine variant",
    )
    return parser.parse_args()


def _load_results(path: Path) -> list[dict]:
    with path.open() as file:
        results = json.load(file)
    if not isinstance(results, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return results


def _panel_label(panel: Panel, m: int) -> str:
    suffix = {2048: "2k", 8192: "8k"}.get(m, str(m))
    return f"{panel.model}_{panel.operation}_{suffix}"


def _active_benchmarks(include_gpu_enqueued: bool) -> tuple[tuple[str, str, str], ...]:
    if include_gpu_enqueued:
        return (*BENCHMARKS, GPU_ENQUEUED_BENCHMARK)
    return BENCHMARKS


def _extract_panel_rows(
    results: list[dict],
    panel: Panel,
    m_values: list[int],
    benchmarks: tuple[tuple[str, str, str], ...],
    normalize: bool,
) -> list[dict]:
    by_label = {result.get("label"): result for result in results}
    rows = []
    for m in m_values:
        label = _panel_label(panel, m)
        if label not in by_label:
            raise ValueError(f"Missing result {label!r}")

        benchmark_results = by_label[label].get("benchmarks") or {}
        row = {"M": m}
        for benchmark_key, _, _ in benchmarks:
            benchmark = benchmark_results.get(benchmark_key)
            if not isinstance(benchmark, dict) or not isinstance(benchmark.get("tflops"), (int, float)):
                raise ValueError(f"Missing numeric tflops for {label}.{benchmark_key}")
            row[benchmark_key] = benchmark["tflops"]

        if normalize:
            baseline = row["tritonblas_rcclbaseline"]
            if baseline <= 0:
                raise ValueError(f"Cannot normalize {label}: TritonBLAS TFLOP/s must be positive")
            for benchmark_key, _, _ in benchmarks:
                row[benchmark_key] /= baseline
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict], benchmarks: tuple[tuple[str, str, str], ...]) -> None:
    fieldnames = ["M", *(benchmark[0] for benchmark in benchmarks)]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(
    path: Path,
    panel_rows: list[tuple[Panel, list[dict]]],
    m_values: list[int],
    benchmarks: tuple[tuple[str, str, str], ...],
    normalized: bool,
) -> None:
    y_label = r"Speedup vs. TritonBLAS" if normalized else r"TFLOP/s"
    lines = [
        r"\documentclass[tikz,border=3pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{tritonblue}{HTML}{4C78A8}",
        r"\definecolor{twoshotorange}{HTML}{F58518}",
        r"\definecolor{copygreen}{HTML}{54A24B}",
        r"\definecolor{enqueuepurple}{HTML}{B279A2}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 2, horizontal sep=0.75cm, vertical sep=1.05cm},",
        r"  width=0.31\textwidth, height=0.27\textwidth,",
        r"  ybar, ymin=0,",
        "  symbolic x coords={" + ",".join(str(m) for m in m_values) + "},",
        r"  xtick=data, enlarge x limits=0.22,",
        r"  grid=major, major grid style={gray!25}, axis on top,",
        r"  tick label style={font=\small}, label style={font=\small}, title style={font=\small},",
        rf"  legend style={{font=\small, draw=none, legend columns={len(benchmarks)}}},",
        r"]",
    ]

    color_names = ("tritonblue", "twoshotorange", "copygreen", "enqueuepurple")
    for panel_index, (panel, _) in enumerate(panel_rows):
        axis_options = [f"title={{{panel.title}}}"]
        if panel_index % 3 == 0:
            axis_options.append(f"ylabel={{{y_label}}}")
        if panel_index >= 3:
            axis_options.append(r"xlabel={$M$}")
        if panel_index == 0:
            axis_options.append("legend to name=figurelegend")
        lines.append(r"\nextgroupplot[" + ", ".join(axis_options) + "]")

        relative_csv = f"data/{panel.csv_name}"
        if normalized:
            reference_coordinates = " ".join(f"({m},1)" for m in m_values)
            lines.append(
                rf"\addplot+[black!60, dashed, thick, mark=none, sharp plot, forget plot] "
                rf"coordinates {{{reference_coordinates}}};"
            )
        for benchmark_index, (benchmark_key, display_name, _) in enumerate(benchmarks):
            color = color_names[benchmark_index]
            lines.append(
                rf"\addplot+[fill={color}, draw={color}, bar width=5pt, area legend] "
                rf"table[x=M,y={benchmark_key},col sep=comma] {{{relative_csv}}};"
            )
            if panel_index == 0:
                lines.append(rf"\addlegendentry{{{display_name}}}")

    lines.extend(
        [
            r"\end{groupplot}",
            r"\node[anchor=north] at ([yshift=-0.55cm]group c2r2.south) {\ref{figurelegend}};",
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _plot_matplotlib(
    output_dir: Path,
    panel_rows: list[tuple[Panel, list[dict]]],
    m_values: list[int],
    benchmarks: tuple[tuple[str, str, str], ...],
    normalized: bool,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required for PDF/PNG output. Run with benchmark/ops/env/bin/python."
        ) from error

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    x_positions = list(range(len(m_values)))
    bar_width = 0.8 / len(benchmarks)

    legend_handles = None
    legend_labels = None
    for axis, (panel, rows) in zip(axes.flat, panel_rows):
        for benchmark_index, (benchmark_key, display_name, color) in enumerate(benchmarks):
            offset = (benchmark_index - (len(benchmarks) - 1) / 2) * bar_width
            axis.bar(
                [x + offset for x in x_positions],
                [row[benchmark_key] for row in rows],
                width=bar_width,
                label=display_name,
                color=color,
            )
        axis.set_title(panel.title)
        axis.set_xticks(x_positions, labels=[str(m) for m in m_values])
        axis.set_axisbelow(True)
        axis.grid(True, axis="y", alpha=0.25)
        if normalized:
            axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1.5, zorder=1)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    y_label = "Speedup vs. TritonBLAS + RCCL" if normalized else "TFLOP/s"
    for axis in axes[:, 0]:
        axis.set_ylabel(y_label)
    for axis in axes[-1, :]:
        axis.set_xlabel("M")

    figure.legend(legend_handles, legend_labels, loc="outside upper center", ncols=len(benchmarks), frameon=False)
    figure.savefig(output_dir / "matmul_all_reduce_six_panel.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "matmul_all_reduce_six_panel.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if len(set(args.m_values)) != len(args.m_values) or any(m <= 0 for m in args.m_values):
        raise SystemExit("error: --m-values must contain unique positive integers")

    benchmarks = _active_benchmarks(args.include_gpu_enqueued)
    plotted_benchmarks = benchmarks[1:] if args.normalize_to_tritonblas else benchmarks
    try:
        results = _load_results(args.input)
        panel_rows = [
            (
                panel,
                _extract_panel_rows(
                    results,
                    panel,
                    args.m_values,
                    benchmarks,
                    args.normalize_to_tritonblas,
                ),
            )
            for panel in PANELS
        ]
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for panel, rows in panel_rows:
        _write_csv(data_dir / panel.csv_name, rows, benchmarks)
    _write_tex(
        args.output_dir / "matmul_all_reduce_six_panel.tex",
        panel_rows,
        args.m_values,
        plotted_benchmarks,
        args.normalize_to_tritonblas,
    )

    try:
        _plot_matplotlib(
            args.output_dir,
            panel_rows,
            args.m_values,
            plotted_benchmarks,
            args.normalize_to_tritonblas,
            args.dpi,
        )
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error

    print(f"Wrote six-panel figure, TeX source, and CSV data to {args.output_dir}")


if __name__ == "__main__":
    main()
