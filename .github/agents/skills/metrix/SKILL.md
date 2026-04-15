---
name: metrix-profiling
description: Profile GPU kernels when performance analysis or optimization is required. Use for AMD ROCm GPU metrics, bandwidth, cache hit rates, coalescing, or kernel timing.
---

# Metrix: GPU Profiling

Profile AMD GPU kernels and get human-readable metrics (bandwidth, cache, coalescing, FLOPS). Architecture is auto-detected.

## When to Use

- User asks to profile a GPU application or kernel
- Performance analysis, optimization, or bottleneck investigation
- Need HBM/L2/L1 bandwidth, hit rates, or compute metrics
- Need timing-only runs (fast, no hardware counters)

## Instructions

1. **Ensure the target runs on AMD ROCm** (e.g. `hipcc`-built binary or Python script that launches HIP/ROCm kernels).
2. **Choose execution path:**
   - If a Metrix MCP server is available, use its profile tool with the same options below.
   - Otherwise run the CLI or Python API from the environment where Metrix is installed.

### CLI

From the project or install prefix:

```bash
# Profile with all metrics (auto-detected arch)
metrix ./my_app

# Time only (fast, no counters)
metrix --time-only -n 10 ./my_app

# Filter kernels by name
metrix --kernel matmul ./my_app

# Specific metrics
metrix --metrics memory.l2_hit_rate,memory.coalescing_efficiency,compute.total_flops ./my_app

# Save to JSON/CSV
metrix -o results.json ./my_app
```

Options: `--profile`/`-p` (run `metrix list profiles` for names: `quick`, `memory`, `memory_bandwidth`, `memory_cache`, `compute`), `--metrics`/`-m`, `--time-only`, `--kernel`/`-k` (regular expression), `--num-replays`/`-n`, `--output`/`-o`, `--top`, `--aggregate`, `--timeout`, `--no-counters`, `--log`/`-l`, `--quiet`/`-q`. Discovery: `metrix list <metrics|profiles|devices>`, `metrix info <metric|profile> <name>`. Note: `metrix list counters` and `metrix info counter <name>` are not implemented yet (CLI reports “not yet implemented”).

### Python API

```python
from metrix import Metrix

profiler = Metrix()
results = profiler.profile("./my_app", num_replays=5)

for kernel in results.kernels:
    print(kernel.name, kernel.duration_us.avg)
    for metric, stats in kernel.metrics.items():
        print(f"  {metric}: {stats.avg}")
```

Use `metrics=[...]` for a subset; omit for all metrics. Use `cwd` when the binary expects a specific working directory.

## Workflow

1. Identify the executable or script to profile (e.g. `./app` or `python run_kernels.py`).
2. If only timing is needed, use `--time-only` for speed.
3. If full metrics are needed, run `metrix ./app` (or MCP equivalent); optionally restrict with `--kernel` or `--metrics`.
4. Interpret results: low L2 hit rate, low coalescing, or low HBM utilization suggest optimization targets.
5. For automation or tooling, use `-o results.json` and parse the JSON output.

## Key Metrics (reference)

- **Memory:** `memory.hbm_bandwidth_utilization`, `memory.l2_hit_rate`, `memory.l1_hit_rate`, `memory.coalescing_efficiency`, `memory.global_load_efficiency`, `memory.lds_bank_conflicts`, `memory.atomic_latency`
- **Compute:** `compute.total_flops`, `compute.hbm_gflops`, `compute.hbm_arithmetic_intensity`, `compute.l2_arithmetic_intensity`, `compute.l1_arithmetic_intensity`

Use relative paths for the target binary and output files so the skill is portable across environments.
