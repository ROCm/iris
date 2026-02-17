# Benchmarking Harness (iris.bench)

The `iris.bench` module provides a unified, decorator-based infrastructure for benchmarking Iris operations.

## Overview

The benchmarking harness eliminates code duplication by providing:

- **Automatic iris instance management**: The decorator creates and manages the iris instance
- **Code organization**: Use @setup, @preamble, @measure annotations
- **Automatic statistics**: mean, median, p50, p99, min, max
- **Barrier synchronization**: Built-in multi-GPU support
- **Structured output**: JSON export for CI/CD

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

## API Reference

### @benchmark Decorator

Main decorator for benchmarking with automatic iris instance management.

```python
@benchmark(
    name: str,
    warmup: int = 25,
    iters: int = 100,
    heap_size: int = 1 << 33,
    auto_print: bool = False,
)
```

**Parameters:**
- `name` - Benchmark name
- `warmup` - Number of warmup iterations (default: 25)
- `iters` - Number of timing iterations (default: 100)
- `heap_size` - Iris symmetric heap size (default: 1<<33)
- `auto_print` - Automatically print results (default: False)

**Returns:** BenchmarkResult

### Code Annotations

Within your benchmark function, use these decorators to organize code:

#### @setup
Runs **once** before any timing starts.

**Use for:**
- Tensor allocation
- Initial data setup
- One-time configuration

**Returns:** Values passed to @preamble and @measure

#### @preamble
Runs **before each timed iteration**.

**Use for:**
- Resetting output buffers
- Clearing flags/state
- Per-iteration setup

**Parameters:** Receives values from @setup

#### @measure (Required)
The code that gets **timed**.

**Use for:**
- Kernel launches
- The operation you want to benchmark

**Parameters:** Receives values from @setup

### BenchmarkResult

Dataclass storing benchmark results.

**Attributes:**
- `name: str` - Benchmark name
- `mean_ms: float` - Mean time in milliseconds
- `median_ms: float` - Median time
- `p50_ms: float` - 50th percentile
- `p99_ms: float` - 99th percentile
- `min_ms: float` - Minimum time
- `max_ms: float` - Maximum time
- `n_warmup: int` - Number of warmup iterations
- `n_repeat: int` - Number of timing iterations
- `params: Dict` - Benchmark parameters
- `raw_times: List[float]` - Raw timing measurements

**Methods:**
- `to_dict(include_raw_times=False)` - Convert to dictionary
- `to_json(include_raw_times=False, indent=2)` - Convert to JSON
- `print_summary()` - Print formatted summary

### Utility Functions

#### torch_dtype_from_str

```python
dtype = torch_dtype_from_str("fp16")  # -> torch.float16
```

Supported: `"int8"`, `"fp16"`, `"bf16"`, `"fp32"`

#### compute_bandwidth_gbps

```python
bandwidth = compute_bandwidth_gbps(total_bytes, time_ms)
```

Computes bandwidth in GiB/s.

## Examples

### Example 1: Simple Benchmark

```python
from iris.bench import benchmark

@benchmark(name="vector_add", warmup=5, iters=50)
def bench_add(shmem, size=1024):
    
    @setup
    def allocate():
        a = shmem.randn(size)
        b = shmem.randn(size)
        c = shmem.zeros(size)
        return a, b, c
    
    @measure
    def compute(a, b, c):
        c.copy_(a + b)

result = bench_add(size=1024)
result.print_summary()
```

### Example 2: With Preamble

```python
@benchmark(name="gemm", warmup=5, iters=50, heap_size=1<<33)
def bench_gemm(shmem, m=8192, n=4608, k=36864):
    
    @setup
    def allocate():
        A = shmem.randn(m, k, dtype=torch.float16)
        B = shmem.randn(k, n, dtype=torch.float16)
        C = shmem.zeros(m, n, dtype=torch.float16)
        return A, B, C
    
    @preamble
    def reset(A, B, C):
        C.zero_()
    
    @measure
    def compute(A, B, C):
        gemm_kernel[grid](A, B, C, m, n, k)

result = bench_gemm()
```

### Example 3: Bandwidth Calculation

```python
from iris.bench import benchmark, compute_bandwidth_gbps

@benchmark(name="copy", warmup=5, iters=50)
def bench_copy(shmem, size=1024*1024*256):
    
    @setup
    def allocate():
        src = shmem.randn(size, dtype=torch.float16)
        dst = shmem.zeros(size, dtype=torch.float16)
        return src, dst
    
    @measure
    def copy(src, dst):
        dst.copy_(src)

result = bench_copy()

# Compute bandwidth
element_size = 2  # float16
total_bytes = size * element_size
bandwidth = compute_bandwidth_gbps(total_bytes, result.mean_ms)
print(f"Bandwidth: {bandwidth:.2f} GiB/s")
```

### Example 4: JSON Export

```python
result = bench_gemm(m=8192, n=4608, k=36864)

# Export to JSON
with open("results.json", "w") as f:
    f.write(result.to_json(include_raw_times=True))

# Or use to_dict for custom processing
data = result.to_dict()
print(f"Mean: {data['mean_ms']:.2f} ms")
```

## Integration

The harness uses `iris.do_bench` internally for timing, ensuring consistency with existing code. The @benchmark decorator:
- Creates the iris instance
- Manages barrier synchronization automatically
- Handles warmup and iteration loops
- Computes statistics automatically

## Notes

- The `shmem` parameter is automatically injected by the decorator
- `@setup`, `@preamble`, and `@measure` are injected at runtime
- At least one `@measure` decorated function is required
- `@setup` and `@preamble` are optional

## See Also

- [Quick Start Guide](README_bench.md)
- [Migration Examples](bench_migration_example.md)
- [Working Examples](../examples/benchmark/bench_harness_example.py)
