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
