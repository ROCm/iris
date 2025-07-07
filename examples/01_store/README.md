<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Store benchmark

Store benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/01_store/store_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional STORE bandwidth GiB/s [Remote write]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   1915.78     46.44     46.24     45.49     45.60     45.66     45.02     44.89
GPU 01  ->     46.22   1881.97     45.80     46.13     45.89     46.06     45.36     44.85
GPU 02  ->     46.30     46.11   1886.94     45.82     45.10     44.86     44.97     44.59
GPU 03  ->     45.59     45.88     46.08   1837.94     45.26     45.26     44.71     44.93
GPU 04  ->     45.85     46.21     44.97     45.31   1853.58     45.98     46.09     46.33
GPU 05  ->     46.04     45.97     45.16     45.50     46.28   1862.34     46.28     45.91
GPU 06  ->     45.52     45.36     44.70     44.88     45.85     46.02   1886.26     45.62
GPU 07  ->     45.23     45.27     44.68     44.68     46.21     45.85     45.56   1857.54
```
