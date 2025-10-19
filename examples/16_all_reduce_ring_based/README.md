<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Ring-Based All-Reduce

This example demonstrates a standalone ring-based all-reduce collective operation across multiple GPUs. The ring-based all-reduce is an efficient communication pattern that reduces data across all GPUs by forming a logical ring topology.

In this pattern, each GPU sends data to its neighbor while receiving from the other neighbor, completing the reduction in multiple passes around the ring. This approach provides excellent bandwidth utilization and scales well with the number of GPUs.

## Usage

### Basic Run

To run the benchmark with default parameters:

```terminal
python examples/16_all_reduce_ring_based/benchmark.py --num_ranks 8
```

### Validation

To verify numerical correctness:

```terminal
python examples/16_all_reduce_ring_based/benchmark.py --validate --num_ranks 8
```

### Benchmarking

To run performance benchmarks:

```terminal
python examples/16_all_reduce_ring_based/benchmark.py --benchmark --validate --num_ranks 8
```

### Custom Matrix Dimensions

You can specify custom dimensions for the data to reduce:

```terminal
python examples/16_all_reduce_ring_based/benchmark.py --num_ranks 8 -m 8192 -n 4608
```

### Options

- `-m`: Number of rows in input/output matrix (default: 8192)
- `-n`: Number of columns in input/output matrix (default: 4608)
- `--datatype`: Data type for computation (`fp16`, `fp32`, `bf16`, `int8`) (default: fp16)
- `--validate`: Enable validation mode
- `--benchmark`: Enable benchmarking mode
- `--BLK_M`, `--BLK_N`: Block sizes for tiling (defaults: 128, 128)
