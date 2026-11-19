# Measuring matmul_all_reduce workspace reuse

The `Matmul All-Reduce Workspace Benchmark` PR workflow compares the base commit
against the exact PR head on the same allocated AMD GPU pool. The base source
is read from a Git archive; both implementations run the **head's benchmark
harness**, dependencies, FP16 inputs and fixed 64x64x32 tile configuration.
Each launch uses a single node with 2, 4 or 8 ranks.

The sweep covers M=128/1024/4096, N=1024 and **local** K=512 on each rank, for
`atomic`, `one_shot`, `two_shot` and an unfused `torch.mm` + RCCL control. It measures
matching-workspace reuse through the public API with managed preparation. It
does not force `workspace.prepared`, bypass workspace agreement, clear flags
manually, or exclude preparation from the measured path.

Each case has five warmups, three trials and twenty timed samples per trial.
Compilation and initial allocation occur before timing. Results are validated
against `torch.mm` + `dist.all_reduce` before and after timing. Every sample
includes collective completion. For each trial the summary takes the slowest
rank's median, then reports the median across trials. Inputs, configurations,
versions, commits and all per-rank samples are retained in the JSON artifacts.

The headline metric is wall-clock latency, including host coordination and
completion on every rank. GPU event elapsed times are also retained, but include
host launch gaps and in-call synchronization: they are **not kernel-only times**.
The workflow publishes a full case table and geometric mean over all fused cases;
it imposes no speedup threshold. A slowdown is a valid result. The unfused control
helps reveal differences in runner load or device state between the two runs.
The job releases its GPUs and uploads completed artifacts even if a benchmark fails.

To run manually on an AMD GPU host, install Iris and its GPU dependencies, then
use the same script for each source tree:

```bash
torchrun --standalone --nproc_per_node=2 \
    benchmark/ops/bench_matmul_all_reduce_workspace.py \
    --source-dir /path/to/base --commit BASE_SHA --label base --output results/base_2.json
torchrun --standalone --nproc_per_node=2 \
    benchmark/ops/bench_matmul_all_reduce_workspace.py \
    --source-dir /path/to/head --commit HEAD_SHA --label head --output results/head_2.json
python benchmark/ops/summarize_workspace_benchmark.py results --output results/summary.md
```

Rerun paired measurements, preferably alternating revision order, before using
a performance number publicly. Include GPU model, rank count, shape, precision,
comparison revision and measurement scope. Report the measured result, including
regressions; do not present the best shape as a general speedup or call stream
envelope measurements kernel latency. Passing tests establishes the covered
correctness cases, not a performance improvement. These experiments do not
validate CUDA graph replay or concurrent use of a workspace on multiple streams.
