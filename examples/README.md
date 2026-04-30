**[README](../README.md)** » **Examples**

# Examples

This directory contains examples organized by API tier, progressing from low-level device primitives to high-level fused operations.

## Running Examples

All examples use `torchrun` for multi-GPU execution:

```bash
torchrun --nproc_per_node=<num_gpus> --standalone examples/<name>/example.py [--validate]
```

## Directory Structure

### Host APIs

| # | Example | API | Description |
|---|---------|-----|-------------|
| 00 | [`00_heap_basics`](00_heap_basics) | `ctx.zeros`, `ctx.ones`, `ctx.randn`, `ctx.is_symmetric`, `ctx.as_symmetric` | Symmetric heap allocation and tensor management |
| 01 | [`01_barrier`](01_barrier) | `ctx.barrier`, `ctx.device_barrier` | Host and device barrier synchronization |
| 02 | [`02_broadcast`](02_broadcast) | `ctx.broadcast` | Scalar broadcast from one rank to all |

### iris.mem — Device-Side RMA Primitives

| # | Example | API | Description |
|---|---------|-----|-------------|
| 03 | [`03_mem_load_store`](03_mem_load_store) | `iris.load`, `iris.store` | Remote load and store via Triton kernels |
| 04 | [`04_mem_put_get`](04_mem_put_get) | `iris.put`, `iris.get` | One-sided put and get operations |
| 05 | [`05_mem_atomics`](05_mem_atomics) | `iris.atomic_add`, `iris.atomic_cas`, `iris.atomic_xchg` | Remote atomic operations with memory ordering |
| 06 | [`06_mem_message_passing`](06_mem_message_passing) | `iris.store`, `iris.atomic_cas` | Flag-based producer-consumer pattern |
| 07 | [`07_mem_context`](07_mem_context) | `iris.mem.triton.Context` | OO wrapper that eliminates heap_bases boilerplate |

### GEMM Patterns — Fused Matrix Multiplication + Communication

These examples demonstrate different strategies for overlapping GEMM computation with inter-GPU communication. See the [taxonomy](../docs/conceptual/taxonomy.md) for a detailed comparison.

| # | Example | Pattern | Description |
|---|---------|---------|-------------|
| 08 | [`08_gemm_all_scatter`](08_gemm_all_scatter) | Fused Sequential | GEMM tiles scatter results as they complete |
| 09 | [`09_gemm_all_reduce_atomics`](09_gemm_all_reduce_atomics) | Fused Atomic | K-split GEMM with atomic all-reduce |
| 10 | [`10_gemm_one_shot_all_reduce`](10_gemm_one_shot_all_reduce) | Fused One-Shot | K-split GEMM with one-shot all-reduce |
| 11 | [`11_gemm_all_scatter_wg_specialization`](11_gemm_all_scatter_wg_specialization) | Fused P-C (WG Spec) | Workgroup specialization for compute vs. communication |
| 12 | [`12_gemm_all_scatter_producer_consumer`](12_gemm_all_scatter_producer_consumer) | Unfused P-C | Concurrent GEMM + scatter kernels |
| 13 | [`13_gemm_all_scatter_bulk_synchronous`](13_gemm_all_scatter_bulk_synchronous) | Unfused Bulk Sync | Separate GEMM and scatter phases |
| 14 | [`14_gemm_all_reduce_ring_based`](14_gemm_all_reduce_ring_based) | Ring All-Reduce | GEMM with ring-based all-reduce |
| 15 | [`15_gemm_all_scatter_independent`](15_gemm_all_scatter_independent) | Independent | Independent GEMM + all-scatter with CSV sweep |
| 16 | [`16_gemm_one_shot_all_reduce_independent`](16_gemm_one_shot_all_reduce_independent) | Independent | Independent GEMM + all-reduce with CSV sweep |
| 17 | [`17_gemm_one_shot_reduce_scatter_wg_specialization`](17_gemm_one_shot_reduce_scatter_wg_specialization) | Fused RS (WG Spec) | GEMM + reduce-scatter with workgroup specialization |
| 18 | [`18_gemm_all_scatter_tracing`](18_gemm_all_scatter_tracing) | Fused + Tracing | GEMM + all-scatter with tile-level tracing |

### Advanced Examples

| # | Example | Description |
|---|---------|-------------|
| 19 | [`19_flash_decode`](19_flash_decode) | Fused Flash Decode Attention for LLM inference |
| 20 | [`20_all_gather_gemm`](20_all_gather_gemm) | Fused All-Gather + GEMM (pull and push models) |
| 21 | [`21_all_reduce_ring_based`](21_all_reduce_ring_based) | Standalone ring-based all-reduce |
| 22 | [`22_expert_sharded_moe`](22_expert_sharded_moe) | Expert-sharded Mixture of Experts |
| 23 | [`23_message_passing`](23_message_passing) | Point-to-point message passing with device context |
| 24 | [`24_gluon_all_gather_tracing`](24_gluon_all_gather_tracing) | All-gather with Gluon backend tracing |

### iris.ccl — Host-Side Collective Communication

| # | Example | API | Description |
|---|---------|-----|-------------|
| 25 | [`25_ccl_all_reduce`](25_ccl_all_reduce) | `ctx.ccl.all_reduce` | Sum-reduce tensors across all ranks |
| 26 | [`26_ccl_all_gather`](26_ccl_all_gather) | `ctx.ccl.all_gather` | Gather tensors from all ranks |
| 27 | [`27_ccl_all_to_all`](27_ccl_all_to_all) | `ctx.ccl.all_to_all` | Transpose data across all ranks |
| 28 | [`28_ccl_reduce_scatter`](28_ccl_reduce_scatter) | `ctx.ccl.reduce_scatter` | Reduce and scatter across ranks |

### iris.x — Tile-Level Collectives

These examples use the `iris.x` tile abstraction for fine-grained control over collective communication within Triton kernels.

| # | Example | API | Description |
|---|---------|-----|-------------|
| 29 | [`29_x_all_reduce`](29_x_all_reduce) | `iris.x.all_reduce_atomic` | Tile-level atomic all-reduce |
| 30 | [`30_x_all_gather`](30_x_all_gather) | `iris.x.all_gather` | Tile-level all-gather |
| 31 | [`31_x_reduce_scatter`](31_x_reduce_scatter) | `iris.x.reduce_scatter` | Tile-level reduce-scatter with locks |
| 32 | [`32_x_all_to_all`](32_x_all_to_all) | `iris.x.all_to_all` | Tile-level all-to-all |
| 33 | [`33_x_gather`](33_x_gather) | `iris.x.gather` | Tile-level gather from a specific rank |
| 34 | [`34_x_gemm_all_reduce`](34_x_gemm_all_reduce) | `iris.x.all_reduce_atomic` + `tl.dot` | Fused GEMM + tile-level all-reduce |

### iris.ops — Fused Operations

| # | Example | API | Description |
|---|---------|-----|-------------|
| 35 | [`35_ops_matmul_all_reduce`](35_ops_matmul_all_reduce) | `ctx.ops.matmul_all_reduce` | Fused GEMM + all-reduce |
| 36 | [`36_ops_all_gather_matmul`](36_ops_all_gather_matmul) | `ctx.ops.all_gather_matmul` | Fused all-gather + GEMM |
| 37 | [`37_ops_matmul_all_gather`](37_ops_matmul_all_gather) | `ctx.ops.matmul_all_gather` | Fused GEMM + all-gather |
| 38 | [`38_ops_matmul_reduce_scatter`](38_ops_matmul_reduce_scatter) | `ctx.ops.matmul_reduce_scatter` | Fused GEMM + reduce-scatter |

