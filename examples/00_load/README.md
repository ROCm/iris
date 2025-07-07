<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Load benchmark

Load benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/00_load/load_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional LOAD bandwidth GiB/s [Remote read]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   5563.42     47.73     47.52     47.02     46.94     47.42     46.84     46.43
GPU 01  ->     47.54   5154.41     47.21     47.62     47.43     47.08     46.91     46.74
GPU 02  ->     47.54     47.31   5187.24     46.86     46.31     46.57     46.10     45.72
GPU 03  ->     46.97     47.18     47.30   4803.27     46.97     46.79     45.97     45.71
GPU 04  ->     47.43     47.27     46.46     46.59   5091.24     47.48     47.38     47.09
GPU 05  ->     47.34     47.09     46.45     47.11     47.77   5076.19     47.32     47.33
GPU 06  ->     46.98     46.72     46.04     46.11     47.30     47.36   5332.80     46.99
GPU 07  ->     46.02     46.90     45.95     45.95     47.45     47.48     47.32   4798.39
```