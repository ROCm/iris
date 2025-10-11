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
- [`13_flash_decode`](13_flash_decode): Fused Flash Decode Attention for accelerating LLM inference
- [`14_all_gather_gemm`](14_all_gather_gemm): Fused All-Gather + GEMM with Pull and Push models

### Utilities
- [`benchmark`](benchmark): Benchmarking utilities and performance testing tools
- [`common`](common): Common utilities and shared code for examples

## Usage

### Basic Operations
```terminal
# Example command to run distributed load operations
python examples/00_load/load_bench.py --num_ranks 8 # Load across GPUs
python examples/02_all_load/all_load_bench.py --num_ranks 8  # Simultaneous load on all GPUs

# Example command to run distributed store operations
python examples/01_store/store_bench.py --num_ranks 8  # Store across GPUs
python examples/03_all_store/all_store_bench.py --num_ranks 8  # Simultaneous store on all GPUs

# Example command to run atomic operations
python examples/04_atomic_add/atomic_add_bench.py --num_ranks 8  # Atomic add across GPUs
python examples/05_atomic_xchg/atomic_xchg_bench.py --num_ranks 8  # Atomic exchange across GPUs

# Example command to run message passing
python examples/06_message_passing/message_passing_put.py --num_ranks 8
python examples/06_message_passing/message_passing_load_store.py --num_ranks 8
```

### GEMM Operations
```terminal
# Example command to run benchmark with all-scatter algorithm
python examples/07_gemm_all_scatter/benchmark.py --benchmark --validate --num_ranks 8

# Example command to run benchmark with all-reduce algorithm
python examples/08_gemm_atomics_all_reduce/benchmark.py --benchmark --validate --num_ranks 8

# Example command to run benchmark with one-shot all-reduce algorithm
python examples/09_gemm_one_shot_all_reduce/benchmark.py --benchmark --validate --num_ranks 8

# Example command to run benchmark with all-scatter and workgroup specialization
python examples/10_gemm_all_scatter_wg_specialization/benchmark.py --benchmark --validate --num_ranks 8

# Example command to run benchmark with all-scatter producer-consumer pattern
python examples/11_gemm_all_scatter_producer_consumer/benchmark.py --benchmark --validate --num_ranks 8

# Example command to run benchmark with all-scatter bulk synchronous approach
python examples/12_gemm_all_scatter_bulk_synchronous/benchmark.py --benchmark --validate --num_ranks 8

# Flash Decode Attention - simple example run
python examples/13_flash_decode/example_run.py --num_ranks 8

# All-Gather + GEMM - Pull model
python examples/14_all_gather_gemm/example_run_pull.py --num_ranks 8

# All-Gather + GEMM - Push model
python examples/14_all_gather_gemm/example_run_push.py --num_ranks 8
```
