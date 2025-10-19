<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Matrix Multiplication with Ring-Based All-Reduce

This example demonstrates a distributed matrix multiplication (GEMM) operation followed by a ring-based all-reduce communication pattern. The implementation uses a persistent kernel approach where GEMM computation and communication are overlapped.

The ring-based all-reduce is an efficient collective operation that reduces data across all GPUs by forming a logical ring topology. Each GPU sends data to its neighbor while receiving from the other neighbor, completing the reduction in multiple passes around the ring.

## Usage

### Basic Run

To run the benchmark with default parameters:

```terminal
python examples/15_gemm_all_reduce_ring_based/benchmark.py --num_ranks 8
```

### Validation

To verify numerical correctness against a PyTorch reference:

```terminal
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 8
```

### Benchmarking

To run performance benchmarks:

```terminal
python examples/15_gemm_all_reduce_ring_based/benchmark.py --benchmark --validate --num_ranks 8
```

### Custom Matrix Dimensions

You can specify custom matrix dimensions:

```terminal
python examples/15_gemm_all_reduce_ring_based/benchmark.py --num_ranks 8 -m 4096 -n 4096 -k 4096
```

### Options

- `-m`: Number of rows in matrix A (default: 8192)
- `-n`: Number of columns in matrix B (default: 4608)
- `-k`: Common dimension between matrices A and B (default: 36864)
- `--datatype`: Data type for computation (`fp16`, `fp32`, `bf16`, `int8`) (default: fp16)
- `--validate`: Enable validation mode
- `--benchmark`: Enable benchmarking mode
- `--BLK_M`, `--BLK_N`, `--BLK_K`: Block sizes for tiling (defaults: 128, 128, 64)
