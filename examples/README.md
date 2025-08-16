**[README](../README.md)** » **Algorithm Implementations**

# Algorithm Implementations

This directory contains various algorithm implementations for distributed computing and matrix operations.

## Directory Structure

### Basic Operations
- [`00_load`](00_load): Load operations across multiple GPUs
- [`01_store`](01_store): Store operations across multiple GPUs
- [`02_all_load`](02_all_load): Load operations where all GPUs load simultaneously
- [`03_all_store`](03_all_store): Store operations where all GPUs store simultaneously
- [`04_atomic_add`](04_atomic_add): Atomic add operations across multiple GPUs
- [`05_atomic_xchg`](05_atomic_xchg): Atomic exchange operations across multiple GPUs

### Communication Patterns
- [06_message_passing](06_message_passing): Point-to-point message passing (load/store and put/get operations)

### GEMM Operations
- [`07_gemm_all_scatter`](07_gemm_all_scatter): Matrix multiplication with all-scatter communication
- [`08_gemm_atomics_all_reduce`](08_gemm_atomics_all_reduce): Matrix multiplication with all-reduce using atomics
- [`09_gemm_one_shot_all_reduce`](09_gemm_one_shot_all_reduce): Matrix multiplication with one-shot all-reduce
- [`10_gemm_all_scatter_wg_specialization`](10_gemm_all_scatter_wg_specialization): Matrix multiplication with all-scatter using workgroup specialization
- [`11_gemm_all_scatter_producer_consumer`](11_gemm_all_scatter_producer_consumer): Matrix multiplication with all-scatter using producer-consumer concurrent kernels
- [`12_gemm_all_scatter_bulk_synchronous`](12_gemm_all_scatter_bulk_synchronous): Matrix multiplication with all-scatter using the bulk synchronous parallel approach

### Utilities
- [`benchmark`](benchmark): Benchmarking utilities and performance testing tools
- [`common`](common): Common utilities and shared code for examples

## Usage

### Basic Operations
```terminal
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
```terminal
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
