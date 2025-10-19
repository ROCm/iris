<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Matrix Multiplication with One-Shot All-Reduce (Producer-Consumer Pattern)

This example demonstrates a distributed matrix multiplication (GEMM) operation with a one-shot all-reduce using a producer-consumer pattern. The implementation explores two distinct distribution modes for managing data communication between GPUs.

## Distribution Modes

The example supports two distribution strategies:

### Mode 0: Striding Distribution
Data is distributed in a strided pattern across GPUs, providing fine-grained interleaving of work.

### Mode 1: Block Distribution  
Data is distributed in contiguous blocks across GPUs, providing coarse-grained partitioning of work.

## Usage

### Basic Run with Striding Distribution

```terminal
python examples/17_gemm_one_shot_all_reduce_pc/benchmark.py --num_ranks 8 --distribution 0
```

### Basic Run with Block Distribution

```terminal
python examples/17_gemm_one_shot_all_reduce_pc/benchmark.py --num_ranks 8 --distribution 1
```

### Validation

To verify numerical correctness with striding distribution:

```terminal
python examples/17_gemm_one_shot_all_reduce_pc/benchmark.py --validate --num_ranks 8 --distribution 0
```

To verify with block distribution:

```terminal
python examples/17_gemm_one_shot_all_reduce_pc/benchmark.py --validate --num_ranks 8 --distribution 1
```

### Benchmarking

To run performance benchmarks:

```terminal
python examples/17_gemm_one_shot_all_reduce_pc/benchmark.py --benchmark --validate --num_ranks 8 --distribution 0
```

### Custom Matrix Dimensions

You can specify custom matrix dimensions:

```terminal
python examples/17_gemm_one_shot_all_reduce_pc/benchmark.py --num_ranks 8 -m 4096 -n 4096 -k 4096 --distribution 0
```

### Options

- `-m`: Number of rows in matrix A (default: 8192)
- `-n`: Number of columns in matrix B (default: 4608)
- `-k`: Common dimension between matrices A and B (default: 36864)
- `--distribution`: Distribution mode (0=striding, 1=block) (default: 0)
- `--datatype`: Data type for computation (`fp16`, `fp32`, `bf16`, `int8`) (default: fp16)
- `--validate`: Enable validation mode
- `--benchmark`: Enable benchmarking mode
- `--BLK_M`, `--BLK_N`, `--BLK_K`: Block sizes for tiling (defaults: 256, 64, 64)
