<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Reference Implementation using NCCL

This directory contains reference implementations of distributed GEMM using different collective communication patterns.

## Algorithms

1. **All-Gather**: Split matrix B column-wise across GPUs, compute partial GEMM, then gather results
2. **All-Reduce**: Split matrix B row-wise across GPUs, compute partial GEMM, then reduce results  
3. **Reduce-Scatter**: Split matrix B column-wise, compute partial GEMM, then reduce-scatter results

## Quick Start

Run any algorithm with 8 GPUs:

```bash
# All-Gather variant
python -m torch.distributed.run --nproc_per_node=8 examples/benchmark/reference/all_gather.py --benchmark --validate -m 8192 -n 4608 -k 36864

# All-Reduce variant  
python -m torch.distributed.run --nproc_per_node=8 examples/benchmark/reference/all_reduce.py --benchmark --validate -m 8192 -n 4608 -k 36864

# Reduce-Scatter variant
python -m torch.distributed.run --nproc_per_node=8 examples/benchmark/reference/reduce_scatter.py --benchmark --validate -m 8192 -n 4608 -k 36864
```

## Options

- `-m, -n, -k`: Matrix dimensions (rows, cols, shared dimension)
- `--benchmark`: Enable performance benchmarking
- `--validate`: Verify correctness against single-GPU result
- `--datatype`: Choose `fp16`, `fp32`, or `bf16` (default: `fp16`)
- `--output_file`: JSON output file for results

## Files

- `all_gather.py` - Column-wise split with all-gather
- `all_reduce.py` - Row-wise split with all-reduce  
- `reduce_scatter.py` - Column-wise split with reduce-scatter
- `bench_all_shapes.py` - Automated benchmarking across multiple shapes
- `gemm.py` - Single-GPU reference implementation
