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
- [`15_gemm_all_reduce_ring_based`](15_gemm_all_reduce_ring_based): Matrix multiplication with ring-based all-reduce
- [`16_all_reduce_ring_based`](16_all_reduce_ring_based): Ring-based all-reduce operation
- [`20_gemm_all_scatter_independent`](20_gemm_all_scatter_independent): Independent GEMM and all-scatter operations with support for CSV input configurations
- [`21_gemm_one_shot_all_reduce_independent`](21_gemm_one_shot_all_reduce_independent): Independent GEMM and all-reduce operations with support for CSV input configurations and selective execution

### Collective Communication Library
- [`ccl`](ccl): iris-ccl collective communication operations (all-to-all, etc.)

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

# Example command to run benchmark with ring-based all-reduce for GEMM
python examples/15_gemm_all_reduce_ring_based/benchmark.py --benchmark --validate --num_ranks 8

# Example command to run benchmark with ring-based all-reduce
python examples/16_all_reduce_ring_based/benchmark.py --benchmark --validate --num_ranks 8

# Independent GEMM and all-scatter - single configuration
python examples/20_gemm_all_scatter_independent/benchmark.py --benchmark --validate --num_ranks 8

# Independent GEMM and all-scatter - sweep with CSV configurations
python examples/20_gemm_all_scatter_independent/benchmark.py --benchmark --validate --num_ranks 8 --csv dataset/gemm_config.csv

# Independent GEMM and all-reduce - run both operations
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --benchmark --validate --num_ranks 8

# Independent GEMM and all-reduce - run only GEMM
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --only_gemm --validate --num_ranks 8

# Independent GEMM and all-reduce - run only all-reduce
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --only_comm --validate --num_ranks 8

# Independent GEMM and all-reduce - sweep with CSV configurations
python examples/21_gemm_one_shot_all_reduce_independent/benchmark.py --benchmark --num_ranks 8 --csv examples/21_gemm_one_shot_all_reduce_independent/example_config.csv

# All-to-all collective communication
python examples/ccl/benchmark.py --benchmark --validate -m 1024 -n 512 -r 8 --datatype fp32
```

### CSV Configuration Format

**Note:** Only examples 20 and 21 support loading multiple configurations from a CSV file using the `--csv` argument.

**Example 20 CSV format:**
```csv
m,n,k,datatype,blk_m,blk_n,blk_k,gemm_sms,comm_sms
8192,4608,36864,fp16,256,64,64,256,48
8192,4096,12288,fp32,256,128,64,256,48
4096,4096,8192,bf16,128,128,64,240,56
```

**Example 21 CSV format:**
```csv
m,n,k,datatype,blk_m,blk_n,blk_k,gemm_sms,comm_sms
8192,4608,36864,fp16,256,64,64,256,48
4096,4096,12288,fp32,128,128,64,240,56
```
