# Examples

We've curated a growing collection of practical examples that showcase the power and flexibility of Iris for distributed computing and matrix operations. From basic memory operations to sophisticated GEMM implementations, there's something here for everyone. And guess what? We're constantly adding more examples as we discover new patterns and optimizations!

## Directory Structure

### Basic Operations
- **[00_load](https://github.com/ROCm/iris/tree/main/examples/00_load)**: Load operations across multiple GPUs
- **[01_store](https://github.com/ROCm/iris/tree/main/examples/01_store)**: Store operations across multiple GPUs
- **[02_all_load](https://github.com/ROCm/iris/tree/main/examples/02_all_load)**: Load operations where all GPUs load simultaneously
- **[03_all_store](https://github.com/ROCm/iris/tree/main/examples/03_all_store)**: Store operations where all GPUs store simultaneously
- **[04_atomic_add](https://github.com/ROCm/iris/tree/main/examples/04_atomic_add)**: Atomic add operations across multiple GPUs
- **[05_atomic_xchg](https://github.com/ROCm/iris/tree/main/examples/05_atomic_xchg)**: Atomic exchange operations across multiple GPUs

### Communication Patterns
- **[06_message_passing](https://github.com/ROCm/iris/tree/main/examples/06_message_passing)**: Point-to-point message passing (load/store and put/get operations)

### GEMM Operations
- **[07_gemm_all_scatter](https://github.com/ROCm/iris/tree/main/examples/07_gemm_all_scatter)**: Matrix multiplication with all-scatter communication
- **[08_gemm_atomics_all_reduce](https://github.com/ROCm/iris/tree/main/examples/08_gemm_atomics_all_reduce)**: Matrix multiplication with all-reduce using atomics
- **[09_gemm_one_shot_all_reduce](https://github.com/ROCm/iris/tree/main/examples/09_gemm_one_shot_all_reduce)**: Matrix multiplication with one-shot all-reduce
- **[10_gemm_all_scatter_wg_specialization](https://github.com/ROCm/iris/tree/main/examples/10_gemm_all_scatter_wg_specialization)**: Matrix multiplication with all-scatter using workgroup specialization
- **[11_gemm_all_scatter_producer_consumer](https://github.com/ROCm/iris/tree/main/examples/11_gemm_all_scatter_producer_consumer)**: Matrix multiplication with all-scatter using producer-consumer concurrent kernels
- **[12_gemm_all_scatter_bulk_synchronous](https://github.com/ROCm/iris/tree/main/examples/12_gemm_all_scatter_bulk_synchronous)**: Matrix multiplication with all-scatter using the bulk synchronous parallel approach
- **[13_flash_decode](https://github.com/ROCm/iris/tree/main/examples/13_flash_decode)**: Fused Flash Decode Attention for accelerating LLM inference
- **[14_all_gather_gemm](https://github.com/ROCm/iris/tree/main/examples/14_all_gather_gemm)**: Fused All-Gather + GEMM with Pull and Push models
- **[15_gemm_all_reduce_ring_based](https://github.com/ROCm/iris/tree/main/examples/15_gemm_all_reduce_ring_based)**: Matrix multiplication with ring-based all-reduce
- **[16_all_reduce_ring_based](https://github.com/ROCm/iris/tree/main/examples/16_all_reduce_ring_based)**: Ring-based all-reduce operation
- **[20_gemm_all_scatter_independent](https://github.com/ROCm/iris/tree/main/examples/20_gemm_all_scatter_independent)**: Independent GEMM and all-scatter operations with support for CSV input configurations
- **[21_gemm_one_shot_all_reduce_independent](https://github.com/ROCm/iris/tree/main/examples/21_gemm_one_shot_all_reduce_independent)**: Independent GEMM and all-reduce operations with support for CSV input configurations and selective execution

### Utilities
- **[benchmark](https://github.com/ROCm/iris/tree/main/examples/benchmark)**: Benchmarking utilities and performance testing tools
- **[common](https://github.com/ROCm/iris/tree/main/examples/common)**: Common utilities and shared code for examples
