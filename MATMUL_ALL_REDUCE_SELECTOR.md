# Matmul All-Reduce Selector

## Scope

This policy selects among the validated all-reduce implementations for an
eight-rank MI300X communicator. It is derived from
`benchmark/ops/model_sweep_results_matmul_all_reduce-all-merged-median.json`.

The selector has two levels:

1. Choose direct RMA or copy-engine communication.
2. If using the copy engine, choose host-posted or GPU-enqueued compact SDMA.

## Global Policy

```python
if M <= 512:
    # Removes duplicate partitioned GEMM work when row shards split GEMM tiles.
    return direct_matmul_all_reduce(config=FusedConfig(all_reduce_variant="two_shot_auto"))

# Larger M is consistently faster with a copy-engine variant in the sweep.
return matmul_all_reduce_copy_engine_auto(...)
```

`two_shot_auto` selects the direct compact layout only when the selected GEMM
`BLOCK_M` spans at least two destination row shards. Otherwise it uses the
legacy direct row-shard path.

## Copy-Engine Policy

```python
if 2048 <= M <= 8192:
    return "two_shot"  # compact host-posted SDMA
if K <= 1024:
    return "two_shot_gpu_init_compact"
return "two_shot"
```

Variant meanings:

```text
two_shot                         compact host-posted SDMA
two_shot_gpu_init_compact        compact GPU-enqueued SDMA
two_shot_row_major               legacy host-posted row-major baseline
two_shot_gpu_init                legacy GPU-enqueued row-major baseline
```

## Evidence

The best copy-engine variant beats direct `two_shot_auto` for every measured
shape at `M >= 2048`:

| M | Copy-engine / direct-auto geomean | Copy-engine wins |
|---:|---:|---:|
| 2048 | 1.172x | 15 / 15 |
| 8192 | 1.185x | 15 / 15 |
| 32768 | 1.125x | 15 / 15 |

At `M <= 512`, direct auto wins more individual shapes even though copy engine
has a near-tied aggregate geometric mean. It avoids copy-engine packet and SDMA
queueing sensitivity when the workload has few useful transfer waves. The
copy-engine selector remains complete for callers that explicitly choose the
copy-engine API: shallow K selects GPU-enqueued compact SDMA, while larger K
selects host-posted compact SDMA.

For `M=32768`, GPU-enqueued compact wins all measured `K <= 512` shapes. The
same shallow-K tendency appears in small-M copy-engine-only comparisons. The
host compact path wins all measured `K > 2048` shapes. Shallow K exposes
communication and benefits from GPU-side ready-wave packet submission; large K
provides enough GEMM work for host-posted compact transfers to overlap.

## Limits

K is a compute-to-communication proxy, not a complete model. N and selected
tile geometry still affect packet count, SDMA-wave count, and reduction cost.
Keep the explicit variants in model sweeps and revise these thresholds when new
hardware, rank counts, or model-shape data is added.
