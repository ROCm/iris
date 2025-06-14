<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Store benchmark

Store benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/p2p/store/store_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional STORE bandwidth GiB/s [Remote write]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   3316.50     45.95     45.60     45.92     45.10     45.32     45.55     45.82
GPU 01  ->     46.03   3296.77     46.07     45.29     44.98     45.15     45.63     45.59
GPU 02  ->     45.81     46.00   3283.86     45.82     44.83     44.31     45.48     45.01
GPU 03  ->     46.13     45.45     45.78   3230.44     44.65     44.86     45.24     45.26
GPU 04  ->     45.12     44.89     44.78     44.84   3313.22     45.78     45.73     46.14
GPU 05  ->     45.45     45.44     44.87     44.84     45.74   3183.25     46.03     45.67
GPU 06  ->     45.79     45.81     45.20     45.39     46.01     45.94   3238.77     45.94
GPU 07  ->     45.90     45.48     45.03     45.40     45.97     45.86     46.02   3316.88
```
