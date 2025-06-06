<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# All-Get benchmark

All-Get benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/p2p/all_get/all_get_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:

```terminal
Total Bandwidth (GiB/s) vs Buffer Size
Buffer Size       GPU 00      GPU 01      GPU 02      GPU 03      GPU 04      GPU 05      GPU 06      GPU 07
1.0MB              78.76       76.27       73.88       72.47       72.78       78.67       74.62       70.44
2.0MB              87.75       86.99       84.73       83.96       82.17       85.89       86.94       81.58
4.0MB             100.06       96.54       98.25       98.44       95.50      100.52       99.51       94.79
8.0MB             107.39      104.26      105.02      105.86      103.70      105.48      105.54      104.47
16.0MB            113.11      107.53      109.69      111.18      107.48      111.85      110.88      107.96
32.0MB            114.95      109.16      110.75      115.68      110.19      114.61      111.92      110.72
64.0MB            118.62      110.58      112.52      118.46      112.05      117.24      113.77      111.95
128.0MB           119.65      111.24      113.55      118.86      112.96      119.04      114.80      112.74
256.0MB           119.94      111.81      113.97      120.23      113.77      120.06      116.29      113.53
512.0MB           119.50      111.66      113.48      120.35      114.35      121.12      116.78      113.92
1024.0MB          119.21      111.55      113.03      120.28      115.16      121.54      117.29      114.47
2048.0MB          118.98      111.35      113.01      119.80      115.36      122.09      117.49      114.56
4096.0MB          118.95      111.65      113.27      119.81      115.52      122.52      118.01      114.63
```

The benchmark measures the bandwidth of each GPU receiving data from all other GPUs. Each GPU performs a get operation from every other GPU in the system, and the total bandwidth is calculated based on the total amount of data received and the time taken.