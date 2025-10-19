<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Independent GEMM and One-Shot All-Reduce Operations

This example demonstrates independent execution of matrix multiplication (GEMM) and one-shot all-reduce communication operations. The implementation allows for selective execution of either operation or both together, providing flexibility for testing and benchmarking.

This example supports loading multiple configurations from a CSV file, enabling automated performance sweeps across different matrix dimensions and parameters.

## Usage

### Basic Run (Both Operations)

To run both GEMM and all-reduce with default parameters:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --num_ranks 8
```

### Run Only GEMM Operation

To execute only the matrix multiplication:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --only_gemm --num_ranks 8
```

### Run Only All-Reduce Operation

To execute only the communication operation:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --only_comm --num_ranks 8
```

### Validation

To verify numerical correctness:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --validate --num_ranks 8
```

### Benchmarking

To run performance benchmarks:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --benchmark --validate --num_ranks 8
```

### CSV Configuration Sweep

To run a sweep of configurations from a CSV file:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --benchmark --num_ranks 8 --csv examples/21_gemm_one_shot_all_reduce_independent/example_config.csv
```

The CSV file should have the following format:
```csv
m,n,k,datatype,blk_m,blk_n,blk_k,gemm_sms,comm_sms
8192,4608,36864,fp16,256,64,64,256,48
4096,4096,12288,fp32,128,128,64,240,56
```

### Custom Matrix Dimensions

You can specify custom matrix dimensions:

```terminal
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --num_ranks 8 -m 4096 -n 4096 -k 4096
```

### Options

- `-m`: Number of rows in matrix A (default: 8192)
- `-n`: Number of columns in matrix B (default: 4608)
- `-k`: Common dimension between matrices A and B (default: 36864)
- `--only_gemm`: Run only GEMM operation (mutually exclusive with `--only_comm`)
- `--only_comm`: Run only all-reduce operation (mutually exclusive with `--only_gemm`)
- `--datatype`: Data type for computation (`fp16`, `fp32`, `bf16`, `int8`) (default: fp16)
- `--validate`: Enable validation mode
- `--benchmark`: Enable benchmarking mode
- `--csv`: Path to CSV file with multiple configurations
- `--BLK_M`, `--BLK_N`, `--BLK_K`: Block sizes for tiling (defaults: 256, 64, 64)
