<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Load benchmark

Load benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/05_atomic_xchg/atomic_xchg_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional ATOMIC_XCHG bandwidth GiB/s [Remote atomic exchange]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->    539.99     44.12     44.10     43.88     44.02     44.06     44.14     43.91
GPU 01  ->     44.14    541.93     44.16     44.16     44.14     44.16     44.13     44.12
GPU 02  ->     44.16     44.16    542.29     44.18     44.11     44.11     44.06     44.06
GPU 03  ->     44.15     44.15     44.17    539.64     44.14     44.17     44.11     44.04
GPU 04  ->     44.15     44.17     44.09     44.14    542.28     44.04     44.18     44.16
GPU 05  ->     44.17     44.16     44.08     44.16     44.17    542.57     44.16     44.16
GPU 06  ->     44.15     44.12     43.90     44.02     44.16     44.16    542.12     44.16
GPU 07  ->     44.01     44.15     44.03     43.89     44.17     44.17     44.16    542.67
```