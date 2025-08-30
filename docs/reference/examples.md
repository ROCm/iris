# Examples Reference

This document provides a comprehensive overview of all the example programs included with Iris, organized by complexity and use case.

## Quick Start Examples

### Basic Load/Store Operations

**File**: `examples/00_load/load_bench.py`

Demonstrates fundamental load/store operations between GPUs.

**Key Features:**
- Basic remote memory access
- Synchronization with barriers
- Memory allocation in symmetric heap

**Usage:**
```bash
mpirun -np 2 python examples/00_load/load_bench.py
```

**What it teaches:**
- How to initialize Iris
- Basic load/store patterns
- Memory management fundamentals

### Store Operations

**File**: `examples/01_store/store_bench.py`

Shows how to store data from one GPU to another.

**Key Features:**
- Store operations with different data types
- Performance benchmarking
- Error handling

**Usage:**
```bash
mpirun -np 2 python examples/01_store/store_bench.py
```

## Advanced Operations

### All Load Operations

**File**: `examples/02_all_load/all_load_bench.py`

Demonstrates loading data from all GPUs to a single GPU.

**Key Features:**
- Multi-GPU data aggregation
- Collective communication patterns
- Performance analysis

**Usage:**
```bash
mpirun -np 4 python examples/02_all_load/all_load_bench.py
```

### All Store Operations

**File**: `examples/03_all_store/all_store_bench.py`

Shows how to distribute data from one GPU to all others.

**Key Features:**
- One-to-many communication
- Broadcast patterns
- Memory efficiency

**Usage:**
```bash
mpirun -np 4 python examples/03_all_store/all_store_bench.py
```

## Atomic Operations

### Atomic Add

**File**: `examples/04_atomic_add/atomic_add_bench.py`

Demonstrates atomic addition operations across GPUs.

**Key Features:**
- Atomic operations for counters
- Race condition prevention
- Performance comparison with non-atomic operations

**Usage:**
```bash
mpirun -np 4 python examples/04_atomic_add/atomic_add_bench.py
```

**Example Pattern:**
```python
@triton.jit
def atomic_add_kernel(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Atomically add 1 to each element
    iris.atomic_add(buffer + offsets, 1, 0, 1, heap_bases_ptr, mask=mask)
```

### Atomic Exchange

**File**: `examples/05_atomic_xchg/atomic_xchg_bench.py`

Shows atomic exchange operations for lock-free programming.

**Key Features:**
- Atomic value swapping
- Lock-free synchronization
- Performance benchmarking

**Usage:**
```bash
mpirun -np 4 python examples/05_atomic_xchg/atomic_xchg_bench.py
```

## Communication Patterns

### Message Passing

**File**: `examples/06_message_passing/message_passing_load_store.py`

Demonstrates point-to-point communication using load/store operations.

**Key Features:**
- Bidirectional communication
- Message queuing patterns
- Flow control

**Usage:**
```bash
mpirun -np 2 python examples/06_message_passing/message_passing_load_store.py
```

**File**: `examples/06_message_passing/message_passing_put.py`

Alternative message passing implementation using put operations.

**Key Features:**
- Put-based communication
- Performance comparison
- Different synchronization strategies

## GEMM with Communication

### All-Scatter Pattern

**File**: `examples/07_gemm_all_scatter/`

Complete matrix multiplication example with all-scatter communication.

**Main Files:**
- `gemm_all_scatter.py`: Core GEMM implementation
- `benchmark.py`: Performance benchmarking
- `matmul_wrapper.py`: Utility functions

**Key Features:**
- Matrix multiplication with communication overlap
- All-scatter communication pattern
- Performance optimization techniques

**Usage:**
```bash
# Run benchmark
mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py --benchmark --validate

# Run specific GEMM
mpirun -np 8 python examples/07_gemm_all_scatter/gemm_all_scatter.py
```

### Atomics with All-Reduce

**File**: `examples/08_gemm_atomics_all_reduce/`

GEMM implementation using atomic operations and all-reduce.

**Key Features:**
- Atomic accumulation
- All-reduce for final results
- Alternative to all-scatter pattern

**Usage:**
```bash
mpirun -np 8 python examples/08_gemm_atomics_all_reduce/benchmark.py
```

### One-Shot All-Reduce

**File**: `examples/09_gemm_one_shot_all_reduce/`

Optimized GEMM with single all-reduce operation.

**Key Features:**
- Minimal communication overhead
- Single synchronization point
- Memory-efficient implementation

**Usage:**
```bash
mpirun -np 8 python examples/09_gemm_one_shot_all_reduce/benchmark.py
```

## Advanced Optimization

### Workgroup Specialization

**File**: `examples/10_gemm_all_scatter_wg_specialization/`

GEMM with specialized workgroup patterns.

**Key Features:**
- Workgroup-level optimization
- Specialized communication patterns
- Performance tuning

**Usage:**
```bash
mpirun -np 8 python examples/10_gemm_all_scatter_wg_specialization/benchmark.py
```

### Producer-Consumer Pattern

**File**: `examples/11_gemm_all_scatter_producer_consumer/`

GEMM using producer-consumer communication pattern.

**Key Features:**
- Asynchronous communication
- Producer-consumer queues
- Overlap computation and communication

**Usage:**
```bash
mpirun -np 8 python examples/11_gemm_all_scatter_producer_consumer/benchmark.py
```

### Bulk Synchronous Pattern

**File**: `examples/12_gemm_all_scatter_bulk_synchronous/`

GEMM with bulk synchronous communication.

**Key Features:**
- Bulk communication operations
- Synchronous execution model
- Simplified synchronization

**Usage:**
```bash
mpirun -np 8 python examples/12_gemm_all_scatter_bulk_synchronous/benchmark.py
```

## Benchmark Suite

### All Shapes Benchmark

**File**: `examples/benchmark/bench_all_shapes.py`

Comprehensive benchmarking across different matrix shapes.

**Key Features:**
- Multiple matrix dimensions
- Performance comparison
- Scalability analysis

**Usage:**
```bash
mpirun -np 8 python examples/benchmark/bench_all_shapes.py
```

### Reference Implementations

**Directory**: `examples/benchmark/reference/`

Reference implementations for comparison.

**Files:**
- `all_gather.py`: All-gather communication
- `all_reduce.py`: All-reduce communication
- `gemm.py`: Basic GEMM implementation
- `reduce_scatter.py`: Reduce-scatter communication
- `bench_all_shapes.py`: Benchmarking framework

## Common Utilities

### Utility Functions

**File**: `examples/common/utils.py`

Common utility functions used across examples.

**Key Functions:**
- Memory allocation helpers
- Performance timing utilities
- Validation functions

### Validation

**File**: `examples/common/validation.py`

Validation utilities for checking correctness.

**Key Functions:**
- Result verification
- Error checking
- Performance validation

## Running Examples

### Basic Execution

Most examples can be run with a simple MPI command:

```bash
# Basic 2-GPU execution
mpirun -np 2 python examples/00_load/load_bench.py

# 8-GPU execution for GEMM examples
mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py
```

### Command Line Options

Many examples support command line arguments:

```bash
# Run with benchmarking
python examples/07_gemm_all_scatter/benchmark.py --benchmark

# Run with validation
python examples/07_gemm_all_scatter/benchmark.py --validate

# Run with specific parameters
python examples/07_gemm_all_scatter/benchmark.py --size 2048 --iterations 100
```

### Environment Variables

Set these for optimal performance:

```bash
# ROCm environment
export ROCR_VISIBLE_DEVICES=0,1,2,3

# MPI environment
export OMPI_ALLOW_RUN_AS_ROOT=1

# Iris environment
export IRIS_HEAP_SIZE=2147483648  # 2GB
```

## Learning Path

### Beginner Level

1. **Start with**: `examples/00_load/load_bench.py`
2. **Learn**: Basic operations and memory management
3. **Practice**: Modify buffer sizes and data types

### Intermediate Level

1. **Study**: `examples/04_atomic_add/atomic_add_bench.py`
2. **Learn**: Atomic operations and synchronization
3. **Practice**: Implement custom atomic patterns

### Advanced Level

1. **Master**: `examples/07_gemm_all_scatter/gemm_all_scatter.py`
2. **Learn**: Complex communication patterns
3. **Practice**: Optimize for your specific use case

## Performance Tips

### Memory Management

- **Heap size**: Choose appropriate heap size for your workload
- **Buffer alignment**: Use aligned buffer sizes for optimal performance
- **Memory reuse**: Reuse buffers when possible

### Communication Optimization

- **Batch operations**: Group related operations
- **Overlap**: Overlap computation and communication
- **Reduce barriers**: Minimize synchronization points

### Kernel Optimization

- **Block size**: Experiment with different block sizes
- **Grid size**: Optimize grid dimensions for your GPU
- **Memory access**: Use coalesced memory access patterns

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**: Reduce heap size or buffer size
2. **"MPI rank errors"**: Ensure MPI rank count matches GPU count
3. **"Import errors"**: Verify Iris installation and dependencies

### Debugging Tips

1. **Use logging**: Enable debug logging with `iris_ctx.debug()`
2. **Check barriers**: Ensure proper synchronization
3. **Validate results**: Use validation functions to check correctness

## Contributing Examples

### Adding New Examples

1. **Follow structure**: Use existing examples as templates
2. **Include documentation**: Add comprehensive docstrings
3. **Add validation**: Include correctness checks
4. **Benchmark**: Provide performance measurements

### Example Guidelines

1. **Clear naming**: Use descriptive file and function names
2. **Error handling**: Include proper error handling
3. **Documentation**: Document all parameters and return values
4. **Testing**: Test with different configurations

---

**Ready to explore these examples? Start with the [Basic Operations Tutorial](../tutorials/basic-operations.md) to understand the fundamentals!**
