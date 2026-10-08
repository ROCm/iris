# CCL benchmarks

Each collective is benchmarked against RCCL over identical shapes, through the
same timing harness, via a `backend` axis:

```bash
python benchmark/ccl/bench_all_reduce.py --benchmark_format=csv --benchmark_out=ar.csv
python benchmark/ccl/plot_sweep_results.py ar.csv --output_dir plots
```

`.github/workflows/iris-ccl-benchmark.yml` runs the full grid -- 2/4/8 ranks, the
shapes below, fp16/bf16/fp8 -- after every merge to `main` and on demand, and
publishes the table and plots. Every axis is a `workflow_dispatch` input, so a
narrower run needs no edits. It runs one process per collective, `M` and dtype
with a 64 GiB heap: the benchmarks do not free symmetric buffers between points,
and `all_to_all` at 8 ranks and `M=65536` alone needs ~30 GiB per dtype.

## The RCCL baseline uses ordinary torch tensors

Not Iris heap memory. The symmetric heap is allocated with the flags RMA
requires, so timing RCCL against it would measure Iris's allocation choice
rather than RCCL. Someone deciding between the two would call RCCL on ordinary
tensors, so that is what is measured.

## Shapes

`_common.py` holds the axes. They follow the triton-shmem harness rather than a
square sweep: collectives here move `(tokens, hidden)` tensors, where the token
count varies over orders of magnitude but the hidden dimension stays in a narrow
band. So `M` and `N` are swept independently and `N` is bounded to
`{1024, 2048, 4096, 8192}`. A square sweep spends most of its budget on shapes
nobody runs.

## How results are aggregated

A collective finishes when its slowest participant does, so timing one rank can
hide a straggler on another GPU. Each rank summarises its samples with a median
(outlier-resistant), the per-rank medians are `all_gather`ed, and the headline
`gpu_time_ms` is the **maximum** — the true collective latency. `min_time_ms` and
`skew_pct` are reported alongside, so a load imbalance is visible rather than
averaged away. Bandwidth and TFLOPs derive from the headline.

This matches the triton-shmem harness. Note it changed `iris.bench` behaviour:
the runner previously reported rank 0's mean.

## Figures

`plot_sweep_results.py` writes one image per collective and dtype,
`<output_dir>/<collective>_<dtype>.png` (e.g. `all_reduce_bf16.png`), plus the
Markdown table. Each image has one row per rank count and three panels against
message size:

| panel | content |
|---|---|
| bandwidth | bus bandwidth (GB/s), semilog-x |
| latency | latency (ms), log-log |
| speedup | RCCL latency / Iris latency, log-log; above the 1.0 line means Iris wins |

The table uses the same speedup definition, one section per collective.

Bandwidth is **bus** bandwidth, not algorithmic: the bench scripts declare
`state.set_bytes((W-1) * bytes)` for all_gather and all_to_all, and
`2 * (W-1)/W * bytes` for all_reduce, which is the convention nccl-tests and
RCCL report. The plotter divides the framework's figure straight through and
does not re-apply a factor.

## What is not covered

`reduce_scatter` has no RCCL comparison here. `iris.ccl.reduce_scatter` is
`(M, N) -> (M, N)` — each rank reduces its assigned tiles and stores locally —
whereas `torch.distributed.reduce_scatter_tensor` is `(W*M, N) -> (M, N)`.
Publishing those side by side would compare two different operations.

`all_to_all` is included with a caveat: Iris splits `(M, N*W)` along dim 1 while
`dist.all_to_all_single` splits along dim 0, so the RCCL path uses `(M*W, N)`.
The communication pattern and total bytes are identical — `W-1` peer messages of
`M*N` elements — only the in-memory layout differs.

fp8 has no RCCL baseline, so it is swept for Iris only and the rccl arm of each
benchmark skips it.

This is not limited to reductions. Measured on MI355X / ROCm 7.2.1 / PyTorch
2.10, RCCL rejects fp8 for *every* collective here. `all_gather` and
`all_to_all` fail with an NCCL data-type error, and `all_reduce` additionally
reports `Unsupported Float8 type for NCCL reduction`. Both the OCP types
(`float8_e4m3fn`, `float8_e5m2`) and the `fnuz` variants behave the same way;
all four allocate on device without trouble, so the limitation is in the
collective layer rather than the dtype.

Consequence for the figures: the fp8 images have bandwidth and latency panels
for Iris only, and no speedup panel, which needs a reference to divide by.
