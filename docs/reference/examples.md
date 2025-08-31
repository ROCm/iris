# Examples

This directory contains various algorithm implementations for distributed computing and matrix operations using Iris.

## Directory Structure

### Basic Operations
- **00_load**: Load operations across multiple GPUs
- **01_store**: Store operations across multiple GPUs
- **02_all_load**: Load operations where all GPUs load simultaneously
- **03_all_store**: Store operations where all GPUs store simultaneously
- **04_atomic_add**: Atomic add operations across multiple GPUs
- **05_atomic_xchg**: Atomic exchange operations across multiple GPUs

### Communication Patterns
- **06_message_passing**: Point-to-point message passing (load/store and put/get operations)

### GEMM Operations
- **07_gemm_all_scatter**: Matrix multiplication with all-scatter communication
- **08_gemm_atomics_all_reduce**: Matrix multiplication with all-reduce using atomics
- **09_gemm_one_shot_all_reduce**: Matrix multiplication with one-shot all-reduce
- **10_gemm_all_scatter_wg_specialization**: Matrix multiplication with all-scatter using workgroup specialization
- **11_gemm_all_scatter_producer_consumer**: Matrix multiplication with all-scatter using producer-consumer concurrent kernels
- **12_gemm_all_scatter_bulk_synchronous**: Matrix multiplication with all-scatter using the bulk synchronous parallel approach

### Utilities
- **benchmark**: Benchmarking utilities and performance testing tools
- **common**: Common utilities and shared code for examples

## Usage

### Basic Operations
```bash
# Example command to run distributed load operations
mpirun -np 8 python examples/00_load/load_bench.py  # Load across GPUs
mpirun -np 8 python examples/02_all_load/all_load_bench.py  # Simultaneous load on all GPUs

# Example command to run distributed store operations
mpirun -np 8 python examples/01_store/store_bench.py  # Store across GPUs
mpirun -np 8 python examples/03_all_store/all_store_bench.py  # Simultaneous store on all GPUs

# Example command to run atomic operations
mpirun -np 8 python examples/04_atomic_add/atomic_add_bench.py  # Atomic add across GPUs
mpirun -np 8 python examples/05_atomic_xchg/atomic_xchg_bench.py  # Atomic exchange across GPUs

# Example command to run message passing
python examples/06_message_passing/message_passing_put.py
python examples/06_message_passing/message_passing_load_store.py
```

### GEMM Operations
```bash
# Example command to run benchmark with all-scatter algorithm
mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py --benchmark --validate

# Example command to run benchmark with all-reduce algorithm
mpirun -np 8 python examples/08_gemm_atomics_all_reduce/benchmark.py --benchmark --validate

# Example command to run benchmark with one-shot all-reduce algorithm
mpirun -np 8 python examples/09_gemm_one_shot_all_reduce/benchmark.py --benchmark --validate

# Example command to run benchmark with all-scatter and workgroup specialization
mpirun -np 8 python examples/10_gemm_all_scatter_wg_specialization/benchmark.py --benchmark --validate

# Example command to run benchmark with all-scatter producer-consumer pattern
mpirun -np 8 python examples/11_gemm_all_scatter_producer_consumer/benchmark.py --benchmark --validate

# Example command to run benchmark with all-scatter bulk synchronous approach
mpirun -np 8 python examples/12_gemm_all_scatter_bulk_synchronous/benchmark.py --benchmark --validate
```

## Example Outputs

### Load Benchmark
On an MI300X, the load benchmark will run on 8 GPUs and print bandwidth measurements:
```
Unidirectional LOAD bandwidth GiB/s [Remote read]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   5563.42     47.73     47.52     47.02     46.94     47.42     46.84     46.43
GPU 01  ->     47.54   5154.41     47.21     47.62     47.43     47.08     46.91     46.74
GPU 02  ->     47.54     47.31   5187.24     46.86     46.31     46.57     46.10     45.72
GPU 03  ->     46.97     47.18     47.30   4803.27     46.97     46.79     45.97     45.71
GPU 04  ->     47.43     47.27     46.46     46.59   5091.24     47.48     47.38     47.09
GPU 05  ->     47.34     47.09     46.45     47.11     47.77   5076.19     47.32     47.33
GPU 06  ->     46.98     46.72     46.04     46.11     47.30     47.36   5332.80     46.99
GPU 07  ->     46.02     46.90     45.95     45.95     47.45     47.48     47.32   4798.39
```

### Atomic Add Benchmark
The atomic add benchmark shows atomic operation performance:
```
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

## Getting Started

Start with the basic operations to understand Iris fundamentals:

1. **Load/Store**: Begin with `00_load` and `01_store` to learn basic memory operations
2. **Atomic Operations**: Try `04_atomic_add` to understand atomic operations
3. **Message Passing**: Explore `06_message_passing` for communication patterns
4. **GEMM**: Move to the GEMM examples for complex distributed computing patterns

Each example includes a README with specific usage instructions and expected outputs.
