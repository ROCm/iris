<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# All-Load benchmark

All-Load benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/02_all_load/all_load_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:

```terminal
Total Bandwidth (GiB/s) vs Buffer Size
Size (MiB) | log2(bytes) | GPU 00 | GPU 01 | GPU 02 | GPU 03 | GPU 04 | GPU 05 | GPU 06 | GPU 07
------------------------------------------------------------------------------------------------
       1.0 |          20 |  93.92 |  99.11 |  97.91 |  97.17 |  96.58 |  94.24 |  97.77 |  95.76
       2.0 |          21 | 128.94 | 129.24 | 131.66 | 130.58 | 129.59 | 129.50 | 129.80 | 133.13
       4.0 |          22 | 128.85 | 129.86 | 128.78 | 129.32 | 129.58 | 130.92 | 130.79 | 133.25
       8.0 |          23 | 132.33 | 135.83 | 133.12 | 136.77 | 135.31 | 134.69 | 135.28 | 135.66
      16.0 |          24 | 133.58 | 136.05 | 134.10 | 136.82 | 134.91 | 137.07 | 136.83 | 134.86
      32.0 |          25 | 131.73 | 136.17 | 132.93 | 138.46 | 134.87 | 133.76 | 136.05 | 135.99
      64.0 |          26 | 132.34 | 140.87 | 132.82 | 142.80 | 138.19 | 134.84 | 136.79 | 138.63
     128.0 |          27 | 135.11 | 138.77 | 134.87 | 142.39 | 137.27 | 136.27 | 137.66 | 137.60
     256.0 |          28 | 135.85 | 139.79 | 136.30 | 143.07 | 136.32 | 138.53 | 140.56 | 136.67
     512.0 |          29 | 135.82 | 140.81 | 136.01 | 143.25 | 137.94 | 139.20 | 140.98 | 138.24
    1024.0 |          30 | 135.71 | 141.09 | 136.43 | 141.84 | 137.64 | 139.32 | 141.37 | 137.62
    2048.0 |          31 | 135.66 | 142.26 | 136.37 | 141.90 | 137.38 | 138.46 | 141.06 | 137.53
    4096.0 |          32 | 135.96 | 142.10 | 136.33 | 142.01 | 137.95 | 138.76 | 141.72 | 137.97
```

The benchmark measures the bandwidth of each GPU receiving data from all other GPUs. Each GPU performs a load operation from every other GPU in the system, and the total bandwidth is calculated based on the total amount of data received and the time taken.