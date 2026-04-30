# Examples

A comprehensive collection of examples covering every tier of the Iris API, from low-level device primitives to high-level fused operations.

All examples use `torchrun` for multi-GPU execution:
```bash
torchrun --nproc_per_node=<num_gpus> --standalone examples/<name>/example.py [--validate]
```

## Host APIs

- **[00_heap_basics](https://github.com/ROCm/iris/tree/main/examples/00_heap_basics)**: Symmetric heap allocation (`ctx.zeros`, `ctx.ones`, `ctx.randn`, `ctx.is_symmetric`, `ctx.as_symmetric`)
- **[01_barrier](https://github.com/ROCm/iris/tree/main/examples/01_barrier)**: Host and device barrier synchronization (`ctx.barrier`, `ctx.device_barrier`)
- **[02_broadcast](https://github.com/ROCm/iris/tree/main/examples/02_broadcast)**: Scalar broadcast from one rank to all (`ctx.broadcast`)

## iris.mem — Device-Side RMA Primitives

- **[03_mem_load_store](https://github.com/ROCm/iris/tree/main/examples/03_mem_load_store)**: Remote load and store via Triton kernels (`iris.load`, `iris.store`)
- **[04_mem_put_get](https://github.com/ROCm/iris/tree/main/examples/04_mem_put_get)**: One-sided put and get operations (`iris.put`, `iris.get`)
- **[05_mem_atomics](https://github.com/ROCm/iris/tree/main/examples/05_mem_atomics)**: Remote atomic operations with memory ordering (`iris.atomic_add`, `iris.atomic_cas`, `iris.atomic_xchg`)
- **[06_mem_message_passing](https://github.com/ROCm/iris/tree/main/examples/06_mem_message_passing)**: Flag-based producer-consumer pattern using `iris.store` + `iris.atomic_cas`
- **[07_mem_context](https://github.com/ROCm/iris/tree/main/examples/07_mem_context)**: OO context wrapper that eliminates heap_bases boilerplate (`iris.mem.triton.Context`)

## GEMM Patterns — Fused Matrix Multiplication + Communication

These examples demonstrate different strategies for overlapping GEMM computation with inter-GPU communication. See the [taxonomy](../conceptual/taxonomy.md) for a detailed comparison.

- **[08_gemm_all_scatter](https://github.com/ROCm/iris/tree/main/examples/08_gemm_all_scatter)**: Fused sequential — GEMM tiles scatter results as they complete
- **[09_gemm_all_reduce_atomics](https://github.com/ROCm/iris/tree/main/examples/09_gemm_all_reduce_atomics)**: K-split GEMM with atomic all-reduce
- **[10_gemm_one_shot_all_reduce](https://github.com/ROCm/iris/tree/main/examples/10_gemm_one_shot_all_reduce)**: K-split GEMM with one-shot all-reduce
- **[11_gemm_all_scatter_wg_specialization](https://github.com/ROCm/iris/tree/main/examples/11_gemm_all_scatter_wg_specialization)**: Fused producer-consumer with workgroup specialization
- **[12_gemm_all_scatter_producer_consumer](https://github.com/ROCm/iris/tree/main/examples/12_gemm_all_scatter_producer_consumer)**: Unfused producer-consumer with concurrent kernels
- **[13_gemm_all_scatter_bulk_synchronous](https://github.com/ROCm/iris/tree/main/examples/13_gemm_all_scatter_bulk_synchronous)**: Unfused bulk synchronous with separate phases
- **[14_gemm_all_reduce_ring_based](https://github.com/ROCm/iris/tree/main/examples/14_gemm_all_reduce_ring_based)**: GEMM with ring-based all-reduce
- **[15_gemm_all_scatter_independent](https://github.com/ROCm/iris/tree/main/examples/15_gemm_all_scatter_independent)**: Independent GEMM + all-scatter with CSV configuration sweep
- **[16_gemm_one_shot_all_reduce_independent](https://github.com/ROCm/iris/tree/main/examples/16_gemm_one_shot_all_reduce_independent)**: Independent GEMM + all-reduce with CSV configuration sweep
- **[17_gemm_one_shot_reduce_scatter_wg_specialization](https://github.com/ROCm/iris/tree/main/examples/17_gemm_one_shot_reduce_scatter_wg_specialization)**: GEMM + reduce-scatter with workgroup specialization
- **[18_gemm_all_scatter_tracing](https://github.com/ROCm/iris/tree/main/examples/18_gemm_all_scatter_tracing)**: GEMM + all-scatter with tile-level tracing

## Advanced Examples

- **[19_flash_decode](https://github.com/ROCm/iris/tree/main/examples/19_flash_decode)**: Fused Flash Decode Attention for LLM inference
- **[20_all_gather_gemm](https://github.com/ROCm/iris/tree/main/examples/20_all_gather_gemm)**: Fused All-Gather + GEMM (pull and push models)
- **[21_all_reduce_ring_based](https://github.com/ROCm/iris/tree/main/examples/21_all_reduce_ring_based)**: Standalone ring-based all-reduce
- **[22_expert_sharded_moe](https://github.com/ROCm/iris/tree/main/examples/22_expert_sharded_moe)**: Expert-sharded Mixture of Experts
- **[23_message_passing](https://github.com/ROCm/iris/tree/main/examples/23_message_passing)**: Point-to-point message passing with device context
- **[24_gluon_all_gather_tracing](https://github.com/ROCm/iris/tree/main/examples/24_gluon_all_gather_tracing)**: All-gather with Gluon backend tracing

## iris.ccl — Host-Side Collective Communication

- **[25_ccl_all_reduce](https://github.com/ROCm/iris/tree/main/examples/25_ccl_all_reduce)**: Sum-reduce tensors across all ranks (`ctx.ccl.all_reduce`)
- **[26_ccl_all_gather](https://github.com/ROCm/iris/tree/main/examples/26_ccl_all_gather)**: Gather tensors from all ranks (`ctx.ccl.all_gather`)
- **[27_ccl_all_to_all](https://github.com/ROCm/iris/tree/main/examples/27_ccl_all_to_all)**: Transpose data across all ranks (`ctx.ccl.all_to_all`)
- **[28_ccl_reduce_scatter](https://github.com/ROCm/iris/tree/main/examples/28_ccl_reduce_scatter)**: Reduce and scatter across ranks (`ctx.ccl.reduce_scatter`)

## iris.x — Tile-Level Collectives

- **[29_x_all_reduce](https://github.com/ROCm/iris/tree/main/examples/29_x_all_reduce)**: Tile-level atomic all-reduce (`iris.x.all_reduce_atomic`)
- **[30_x_all_gather](https://github.com/ROCm/iris/tree/main/examples/30_x_all_gather)**: Tile-level all-gather (`iris.x.all_gather`)
- **[31_x_reduce_scatter](https://github.com/ROCm/iris/tree/main/examples/31_x_reduce_scatter)**: Tile-level reduce-scatter with locks (`iris.x.reduce_scatter`)
- **[32_x_all_to_all](https://github.com/ROCm/iris/tree/main/examples/32_x_all_to_all)**: Tile-level all-to-all (`iris.x.all_to_all`)
- **[33_x_gather](https://github.com/ROCm/iris/tree/main/examples/33_x_gather)**: Tile-level gather from a specific rank (`iris.x.gather`)
- **[34_x_gemm_all_reduce](https://github.com/ROCm/iris/tree/main/examples/34_x_gemm_all_reduce)**: Fused GEMM + tile-level all-reduce

## iris.ops — Fused Operations

- **[35_ops_matmul_all_reduce](https://github.com/ROCm/iris/tree/main/examples/35_ops_matmul_all_reduce)**: Fused GEMM + all-reduce (`ctx.ops.matmul_all_reduce`)
- **[36_ops_all_gather_matmul](https://github.com/ROCm/iris/tree/main/examples/36_ops_all_gather_matmul)**: Fused all-gather + GEMM (`ctx.ops.all_gather_matmul`)
- **[37_ops_matmul_all_gather](https://github.com/ROCm/iris/tree/main/examples/37_ops_matmul_all_gather)**: Fused GEMM + all-gather (`ctx.ops.matmul_all_gather`)
- **[38_ops_matmul_reduce_scatter](https://github.com/ROCm/iris/tree/main/examples/38_ops_matmul_reduce_scatter)**: Fused GEMM + reduce-scatter (`ctx.ops.matmul_reduce_scatter`)

