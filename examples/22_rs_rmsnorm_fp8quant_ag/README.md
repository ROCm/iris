<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Reduce-Scatter → RMSNorm → FP8 Quantization → All-Gather benchmark using Iris
This example implements a complete tensor processing pipeline across multiple GPUs:

1. **Reduce-Scatter**: Sum tensors across all GPUs and distribute shards
2. **RMSNorm**: Apply Root Mean Square normalization to each shard
3. **FP8 Quantization**: Quantize to 8-bit floating point (optional)                                                                                                      4. **All-Gather**: Reconstruct the full tensor across all GPUs (optional)

## Usage

```terminal
python benchmark.py --num_rows 8192 --num_cols 7168 --num_ranks 8 --benchmark --fp8_out --all_gather --BLOCK_M 16 --BLOCK_N 64 --num_warps 16 --num_stages 4 --waves_per_eu 4 --rmsnorm_block_size 1024 --rmsnorm_num_warps 8 --rmsnorm_num_prgms 1024 --rmsnorm_waves_per_eu 2 --fp8_block_m 64 --fp8_block_n 64 --fp8_num_warps 4 --fp8_num_stages 2 --fp8_waves_per_eu 2 --ag_block_m 64 --ag_block_n 64 --ag_num_warps 8 --ag_num_stages 3 --ag_waves_per_eu 2 --validate
```

The benchmark measures the bandwidth of each GPU receiving data from all other GPUs. Each GPU performs a load operation from every other GPU in the system, and the total bandwidth is calculated based on the total amount of data received and the time taken.
