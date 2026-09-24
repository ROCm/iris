# CCL benchmarks

Each collective is benchmarked against RCCL over identical shapes, through the
same timing harness, via a `backend` axis:

```bash
python benchmark/ccl/bench_all_reduce.py --benchmark_format=csv --benchmark_out=ar.csv
python benchmark/ccl/plot_sweep_results.py ar.csv --output ccl.png
```

`.github/workflows/iris-ccl-benchmark.yml` runs the sweep on demand and publishes
the table and plots.

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

## What is not covered

`reduce_scatter` has no RCCL comparison here. `iris.ccl.reduce_scatter` is
`(M, N) -> (M, N)` — each rank reduces its assigned tiles and stores locally —
whereas `torch.distributed.reduce_scatter_tensor` is `(W*M, N) -> (M, N)`.
Publishing those side by side would compare two different operations.

`all_to_all` is included with a caveat: Iris splits `(M, N*W)` along dim 1 while
`dist.all_to_all_single` splits along dim 0, so the RCCL path uses `(M*W, N)`.
The communication pattern and total bytes are identical — `W-1` peer messages of
`M*N` elements — only the in-memory layout differs.

fp8 is not swept. RCCL does not implement reductions for the fp8 types, so
`all_reduce` and `reduce_scatter` would have no baseline to compare against;
adding it for the movement-only collectives alone would make the table
inconsistent across operations.
