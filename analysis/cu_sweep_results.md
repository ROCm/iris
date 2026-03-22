# CU Sweep Analysis: All-Gather on MI355X

**Date**: 2026-03-22
**Hardware**: Vultr MI355X (8x MI355X OAM, fully connected XGMI mesh)
**Matrix**: 8192x8192 fp16 (128 MB per rank)
**Methodology**: 10 warmup iterations, 50 timed iterations, CUDA event timing

## Summary

Sweeps `comm_sms` (number of CUs dedicated to communication) to find the
saturation point for XGMI link bandwidth. Compares Triton persistent kernel,
Gluon kernel (with traffic shaping), and RCCL.

## RCCL Baseline

| Ranks | Links | algo_bw (GB/s) | per-link (GB/s) |
|------:|------:|---------------:|----------------:|
|     4 |     3 |         114.25 |           38.08 |
|     8 |     7 |         371.34 |           53.05 |

## 4 Ranks (3 active links)

| CUs | Triton (GB/s) | per-link | Gluon (GB/s) | per-link |
|----:|--------------:|---------:|-------------:|---------:|
|   1 |          9.21 |     3.07 |            — |        — |
|   2 |         17.67 |     5.89 |            — |        — |
|   4 |         34.12 |    11.37 |        19.20 |     6.40 |
|   8 |         66.98 |    22.33 |        38.21 |    12.74 |
|  16 |        130.26 |    43.42 |        75.85 |    25.28 |
|  32 |        153.52 |    51.17 |       148.22 |    49.41 |
|  48 |        158.57 |    52.86 |       150.97 |    50.32 |
|  64 |        162.55 |    54.18 |       161.51 |    53.84 |
|  80 |        163.94 |    54.65 |       163.15 |    54.38 |
|  96 |        165.51 |    55.17 |       159.59 |    53.20 |
| 128 |        167.82 |    55.94 |       161.45 |    53.82 |
| 152 |        166.42 |    55.47 |       165.43 |    55.14 |
| 192 |        167.31 |    55.77 |       164.47 |    54.82 |
| 256 |        168.60 |    56.20 |       163.29 |    54.43 |
| 304 |        162.95 |    54.32 |       163.78 |    54.59 |

### 4-rank comparison at 32 CUs

| Backend | algo_bw (GB/s) | per-link (GB/s) | vs RCCL |
|---------|---------------:|----------------:|--------:|
| RCCL    |         114.25 |           38.08 |   1.00x |
| Gluon   |         148.22 |           49.41 |   1.30x |
| Triton  |         153.52 |           51.17 |   1.34x |

**At 32 CUs, both iris kernels beat RCCL by 30-34%.**

## 8 Ranks (7 active links)

| CUs | Triton (GB/s) | per-link | Gluon (GB/s) | per-link |
|----:|--------------:|---------:|-------------:|---------:|
|   1 |         14.90 |     2.13 |            — |        — |
|   2 |         29.48 |     4.21 |            — |        — |
|   4 |         57.76 |     8.25 |            — |        — |
|   8 |        113.64 |    16.23 |        56.50 |     8.07 |
|  16 |        218.43 |    31.20 |       112.34 |    16.05 |
|  32 |        318.82 |    45.55 |       220.97 |    31.57 |
|  48 |        349.90 |    49.99 |       321.21 |    45.89 |
|  64 |        352.21 |    50.32 |       337.77 |    48.25 |
|  80 |        362.47 |    51.78 |       359.66 |    51.38 |
|  96 |        361.99 |    51.71 |       362.68 |    51.81 |
| 128 |        359.15 |    51.31 |       355.44 |    50.78 |
| 152 |        365.95 |    52.28 |       367.58 |    52.51 |
| 192 |        360.71 |    51.53 |       361.62 |    51.66 |
| 256 |        353.32 |    50.47 |       367.87 |    52.55 |
| 304 |        359.82 |    51.40 |       362.19 |    51.74 |

### 8-rank comparison at 32 CUs

| Backend | algo_bw (GB/s) | per-link (GB/s) | vs RCCL |
|---------|---------------:|----------------:|--------:|
| Gluon   |         220.97 |           31.57 |   0.60x |
| Triton  |         318.82 |           45.55 |   0.86x |
| RCCL    |         371.34 |           53.05 |   1.00x |

### 8-rank comparison at 80 CUs (saturation)

| Backend | algo_bw (GB/s) | per-link (GB/s) | vs RCCL |
|---------|---------------:|----------------:|--------:|
| Gluon   |         359.66 |           51.38 |   0.97x |
| Triton  |         362.47 |           51.78 |   0.98x |
| RCCL    |         371.34 |           53.05 |   1.00x |

## Key Findings

1. **Saturation point**: ~32 CUs for 4 ranks, ~48-80 CUs for 8 ranks.
   Beyond this, adding more CUs gives <5% improvement.

2. **Gluon needs ~2x the CUs of Triton at low CU counts**: At 8 ranks
   with 16 CUs, Triton achieves 218 GB/s vs Gluon 112 GB/s. The gap
   closes by 80 CUs. This is likely due to higher per-CU overhead in the
   gluon kernel (IrisDeviceCtx init, row-by-row iteration, gl.load/ctx.store
   vs Triton's vectorized tl.load/tl.store).

3. **At saturation, all three are within 3%**: At sufficient CUs (80+),
   Triton, Gluon, and RCCL all converge to ~51-53 GB/s per link.

4. **At 32 CUs (practical operating point)**:
   - 4 ranks: iris beats RCCL by 30-34% (RCCL uses more CUs internally)
   - 8 ranks: Triton matches 86% of RCCL, Gluon only 60%

5. **MI355X exceeds MI300X link rates**: We see 52-56 GB/s per link,
   exceeding the 45-48 GB/s documented for MI300X in AMD's RCCL blog.
   Raw link rate is 64 GB/s bidirectional; we achieve 81-88% utilization.

6. **Diminishing returns are steep**: 32->304 CUs (10x) yields only
   ~8% more bandwidth. The XGMI links are the bottleneck, not compute.

---

## Optimization Experiments (2026-03-22)

### Root Cause Analysis

Gluon's per-CU efficiency is lower than Triton because of two structural
differences in the kernel:

1. **Row-by-row iteration**: Gluon iterates `for i in range(BLOCK_SIZE_M)`
   doing 1D loads/stores per row. Triton loads the full 2D tile
   (BLOCK_SIZE_M x BLOCK_SIZE_N) in one shot. This is 32x more
   load/store instructions per tile.

2. **Pointer translation overhead**: Every `ctx.store()` call invokes
   `_translate()`, which does 2x `gl.load(heap_bases)` + pointer
   arithmetic. Per tile: `BLOCK_SIZE_M * (world_size-1)` translations
   = 32 * 7 = 224 extra heap_base loads at 8 ranks.

### Variant: Gluon Hoisted

**Idea**: Pre-compute `local_base = gl.load(ctx.heap_bases + iris_rank)`
once before the tile loop. Inside the inner loop, compute
`target_base = gl.load(ctx.heap_bases + target_iris_rank)` per rank
(compiler should hoist this since `target_iris_rank` is loop-invariant
w.r.t. the row index `i`). Then apply `delta = target_base - local_base`
directly instead of calling `ctx.store()`.

This eliminates the `from_base` load entirely and reduces heap_base loads
from `2 * BLOCK_SIZE_M * (world_size-1)` to `(world_size-1)` per tile.

### Variant: Gluon Partitioned (CU-partitioned)

**Idea**: Assign each CU to one destination rank
(`dest_rank_idx = pid // (COMM_SMS // world_size)`).
Eliminates the inner rank loop. Pre-compute translation delta once per CU.

**Result**: Significantly slower because it breaks data reuse. In the
persistent variant, each CU loads a tile once and writes it to all ranks
(1 load : world_size stores). In the partitioned variant, each CU only
writes to one rank, but the tile data is loaded world_size times total
across all CU groups. The net load:store ratio is the same, but per-CU
data reuse drops from world_size to 1.

### Results: Gluon Variant Comparison

#### 4 Ranks

| CUs | Triton   | Gluon persistent | Gluon hoisted | Gluon partitioned |
|----:|---------:|-----------------:|--------------:|------------------:|
|   8 |    67.68 |            38.23 |     **41.27** |             20.91 |
|  16 |   130.22 |            75.84 |     **81.90** |             42.62 |
|  32 |   152.96 |           148.61 |    **159.81** |             86.13 |
|  48 |   159.07 |           148.57 |    **150.70** |             89.18 |
|  64 |   161.24 |           160.37 |    **161.67** |            103.35 |
|  96 |   165.45 |           160.13 |        159.70 |            121.68 |

#### 8 Ranks

| CUs | Triton   | Gluon persistent | Gluon hoisted | Gluon partitioned |
|----:|---------:|-----------------:|--------------:|------------------:|
|   8 |   112.92 |            56.48 |     **62.51** |             24.01 |
|  16 |   224.01 |           111.97 |    **124.44** |             48.01 |
|  32 |   325.74 |           221.07 |    **243.99** |             96.16 |
|  48 |   344.74 |           318.60 |    **319.42** |            136.95 |
|  64 |   345.60 |           324.34 |    **324.43** |            178.66 |
|  96 |   361.44 |           348.97 |        348.17 |            197.13 |

### Key Takeaways from Optimization

1. **Hoisted translation gives ~8-11% improvement** at low CU counts
   where the overhead matters most. At 32 CUs / 4 ranks, hoisted
   (159.81 GB/s) **beats Triton** (152.96 GB/s) by 4.5%.

2. **CU partitioning is counterproductive** for all-gather. Unlike
   all-to-all (where each CU handles independent src/dst pairs),
   all-gather benefits from the "load once, write everywhere" pattern.
   Partitioning forces each tile to be loaded world_size times.

3. **The remaining gap at 8 ranks** (hoisted 244 vs Triton 326 at 32 CUs)
   is ~1.3x, down from the original ~1.5x. The residual overhead is
   from the row-by-row iteration pattern (32x more instructions than
   Triton's 2D tile loads). Fixing this requires 2D BlockedLayout
   support in the gluon kernel, which gluon supports but has not been
   tested in iris yet.

4. **Ranking**: Triton persistent > Gluon hoisted > Gluon persistent >> Gluon partitioned.
