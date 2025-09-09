<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Load benchmark

Load benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/04_atomic_add/atomic_add_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional ATOMIC_ADD bandwidth GiB/s [Remote atomic add]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->    785.72     15.61     15.64     15.48     15.66     15.58     15.33     15.21
GPU 01  ->     15.68    774.44     15.58     15.65     15.68     15.58     15.32     15.23
GPU 02  ->     15.66     15.62    775.51     15.57     15.16     15.33     15.08     15.15
GPU 03  ->     15.42     15.68     15.59    765.87     15.41     15.50     15.13     15.06
GPU 04  ->     15.58     15.68     15.21     15.32    769.53     15.67     15.58     15.68
GPU 05  ->     15.59     15.49     15.24     15.50     15.57    773.01     15.67     15.59
GPU 06  ->     15.41     15.41     15.15     15.06     15.50     15.67    778.30     15.58
GPU 07  ->     15.22     15.33     15.07     15.06     15.66     15.54     15.56    765.45
```