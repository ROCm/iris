# All-Reduce Benchmark

Benchmark for the Iris all-reduce collective, supporting both single-point runs and
multi-shape sweeps with optional auto-tuning.

## Prerequisites

- 8× AMD MI300X GPUs (or adjust `--num-ranks`)
- ROCm + PyTorch with RCCL backend
- Iris installed or available on `PYTHONPATH`
- `pyyaml` (`pip install pyyaml`) if using `--config`

## Quick Start

All commands assume you are in the Iris root directory:

```bash
cd /home/work/iris
```

### Sweep mode (recommended)

Run all four variants across vLLM-shaped tensors using the provided YAML config:

```bash
HSA_NO_SCRATCH_RECLAIM=1 python3 -u benchmark/ccl/all_reduce/benchmark.py \
  --config benchmark/ccl/all_reduce/configs/vllm_shapes.yaml
```

This sweeps `rccl`, `two_shot`, `ring`, and `one_shot` over decode-like
(M=1,32,64,128,512) and prefill-like (M=2048,4096,8192) sizes with N=2880
and BF16 dtype. Results are printed as a markdown table.

To save results to files:

```bash
HSA_NO_SCRATCH_RECLAIM=1 python3 -u benchmark/ccl/all_reduce/benchmark.py \
  --config benchmark/ccl/all_reduce/configs/vllm_shapes.yaml \
  --json-output results.json \
  --markdown-output results.md
```

### Sweep mode (CLI only, no YAML)

```bash
HSA_NO_SCRATCH_RECLAIM=1 python3 -u benchmark/ccl/all_reduce/benchmark.py \
  --sweep-ms 1,32,128,512,2048,8192 \
  -n 2880 \
  --variants rccl,two_shot,one_shot \
  --datatype bf16 \
  --warmup 50 --iters 200
```

### Single-point mode

Benchmark a single (M, N, variant) configuration, matching the original `benchmark.py`
interface:

```bash
HSA_NO_SCRATCH_RECLAIM=1 python3 -u benchmark/ccl/all_reduce/benchmark.py \
  -m 16384 -n 16384 \
  --variant two_shot \
  --datatype bf16 \
  -b
```

Add `--benchmark-rccl` to include an RCCL comparison, or `-v` to validate correctness:

```bash
HSA_NO_SCRATCH_RECLAIM=1 python3 -u benchmark/ccl/all_reduce/benchmark.py \
  -m 4096 -n 2880 \
  --variant two_shot \
  --datatype bf16 \
  -v -b --benchmark-rccl
```

## Auto-Tuning

Add `--tune` to sweep a config grid (comm_sms, block_size_m, block_size_n, distribution)
per (M, variant) and use the best config for the final measurement:

```bash
HSA_NO_SCRATCH_RECLAIM=1 python3 -u benchmark/ccl/all_reduce/benchmark.py \
  --config benchmark/ccl/all_reduce/configs/vllm_shapes.yaml \
  --tune \
  --tune-warmup 10 --tune-iters 30
```

Tuning adds compilation time on first run (~20-30s per unique constexpr combination)
because `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, and `COMM_SMS` are Triton `tl.constexpr` params.
Compiled kernels are cached by Triton for subsequent runs.

## YAML Config Format

YAML configs live in `configs/`. Example (`configs/vllm_shapes.yaml`):

```yaml
n: 2880
datatype: bf16
num_ranks: 8

variants:
  - rccl
  - two_shot
  - ring
  - one_shot

sweep_ms:
  decode_like:
    - 1
    - 32
    - 64
    - 128
    - 512
  prefill_like:
    - 2048
    - 4096
    - 8192

comm_sms: 64
block_size_m: 64
block_size_n: 64
swizzle_size: 4
distribution: 1
num_rings: 1

warmup: 50
iters: 200
```

The `sweep_ms` keys (`decode_like`, `prefill_like`) become phase labels in the output
tables. CLI flags override any YAML values.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--config` | — | Path to YAML config file |
| `-m` | 16384 | Rows (single-point mode) |
| `-n` | 16384 | Columns |
| `--sweep-ms` | — | Comma-separated M values (enables sweep mode) |
| `--variant` | two_shot | Variant (single-point mode) |
| `--variants` | — | Comma-separated variants (sweep mode) |
| `--datatype` | fp16 | `fp16`, `fp32`, or `bf16` |
| `--comm-sms` | 64 | SMs for comm kernel |
| `--block-size-m` | 64 | Block size M |
| `--block-size-n` | 64 | Block size N |
| `--swizzle-size` | 4 | Swizzle size |
| `--distribution` | 0 | Two-shot distribution (0=strided, 1=block) |
| `--num-rings` | 1 | Ring variant: concurrent rings |
| `--warmup` | 50 | Warmup iterations |
| `--iters` | 200 | Measured iterations |
| `--num-ranks` | 8 | Number of GPU ranks |
| `--tune` | off | Enable auto-tuning |
| `--tune-warmup` | 3 | Tune pass warmup iters |
| `--tune-iters` | 10 | Tune pass measurement iters |
| `-v` | off | Validate correctness (single-point) |
| `-b` | off | Enable benchmarking (single-point) |
| `--benchmark-rccl` | off | Also benchmark RCCL (single-point) |
| `--json-output` | — | JSON output path (sweep mode) |
| `--markdown-output` | — | Markdown output path (sweep mode) |
| `--output-file` | log.json | JSON output (single-point mode) |
| `--init-url` | tcp://127.0.0.1:29527 | Distributed init URL |

## Output

### Sweep mode

Prints a markdown table to stdout with per-variant latency, imbalance, and bandwidth.
Optionally writes JSON and markdown files via `--json-output` / `--markdown-output`.

### Single-point mode

Prints per-rank bandwidth and writes a JSON summary to `--output-file`.

## Troubleshooting

**`EADDRINUSE`**: Change `--init-url` port if re-running shortly after a previous run.

**NCCL timeout with `ring`**: Some ring configs can cause kernel hangs. If this happens,
exclude ring (`--variants rccl,two_shot,one_shot`) or run ring separately with a
different `--init-url` port.

**`HSA_NO_SCRATCH_RECLAIM=1`**: Required on MI300X to avoid scratch memory reclaim issues.
Always set this environment variable.