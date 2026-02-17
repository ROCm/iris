# iris.bench - Unified Benchmarking Harness

A standardized benchmarking infrastructure for Iris using a decorator-based approach.

## Quick Start

```python
from iris.bench import benchmark

@benchmark(name="my_kernel", warmup=5, iters=50)
def run_benchmark(shmem, size=1024):
    # shmem is automatically created by the decorator
    
    @setup
    def allocate():
        buffer = shmem.zeros(size, size)
        return buffer
    
    @measure
    def kernel_launch(buffer):
        my_kernel[grid](buffer)

result = run_benchmark(size=2048)
result.print_summary()
```

## Key Features

- ✅ **Automatic iris instance creation** - The decorator creates and manages the iris instance
- ✅ **Code annotation** - Use @setup, @preamble, and @measure to organize your code
- ✅ **Rich statistics** - mean, median, p50, p99, min, max automatically computed
- ✅ **Automatic barrier synchronization** - Built-in multi-GPU support
- ✅ **JSON export** - Structured results for CI/CD integration
- ✅ **Utility functions** - `torch_dtype_from_str`, `compute_bandwidth_gbps`

## Code Annotations

The benchmarking decorator uses three function annotations:

### @setup
Runs **once** before any timing starts. Use for:
- Tensor allocation
- Initial data setup
- One-time configuration

Returns values are passed to @preamble and @measure functions.

### @preamble
Runs **before each timed iteration**. Use for:
- Resetting output buffers
- Clearing flags/state
- Per-iteration setup

Receives the values returned by @setup.

### @measure
The code that gets **actually timed**. Use for:
- Kernel launches
- The operation you want to benchmark

Receives the values returned by @setup.

## Full Example

```python
from iris.bench import benchmark

@benchmark(name="gemm", warmup=5, iters=50, heap_size=1<<33)
def run_gemm(shmem, m=8192, n=4608, k=36864):
    
    @setup
    def allocate_matrices():
        # Runs once - allocate tensors
        A = shmem.randn(m, k, dtype=torch.float16)
        B = shmem.randn(k, n, dtype=torch.float16)
        C = shmem.zeros(m, n, dtype=torch.float16)
        return A, B, C
    
    @preamble
    def reset_output(A, B, C):
        # Runs before each iteration - clear output
        C.zero_()
    
    @measure
    def compute(A, B, C):
        # This gets timed - run kernel
        gemm_kernel[grid](A, B, C, m, n, k)

result = run_gemm(m=8192, n=4608, k=36864)
result.print_summary()
result.to_json("results.json")  # Export to JSON
```

## Documentation

- 📖 [Full API Documentation](bench_harness.md)
- 📖 [Migration Guide](bench_migration_example.md)
- 💻 [Complete Examples](../examples/benchmark/bench_harness_example.py)

## Testing

```bash
# Run basic tests (no GPU required)
python3 tests/unittests/test_bench_basic.py

# Run full test suite (requires GPU)
pytest tests/unittests/test_bench.py
```

## API Overview

### @benchmark decorator
Main decorator for benchmarking with automatic iris instance management.

**Parameters:**
- `name` - Benchmark name
- `warmup` - Number of warmup iterations (default: 25)
- `iters` - Number of timing iterations (default: 100)
- `heap_size` - Iris heap size (default: 1<<33)
- `auto_print` - Auto-print results (default: False)

### BenchmarkResult
Stores benchmark results with automatic statistics.

**Methods:**
- `print_summary()` - Human-readable output
- `to_dict()` - Convert to dictionary
- `to_json()` - Convert to JSON string

### Utilities
- `torch_dtype_from_str(dtype_str)` - Convert string to torch.dtype
- `compute_bandwidth_gbps(bytes, time_ms)` - Calculate bandwidth

## License

MIT License - Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
