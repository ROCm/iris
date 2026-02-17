# Benchmark Harness Migration Guide

This guide shows how to migrate existing Iris benchmarks to use the new `iris.bench` decorator.

## Key Changes

The new harness:
1. **Decorator-only** - Uses @benchmark decorator exclusively
2. **Automatic iris instance** - Creates and passes `shmem` to your function
3. **Code annotations** - @setup, @preamble, @measure organize your code

## Before: Original Pattern

Original benchmarks had ~100 lines of duplicated boilerplate:

```python
import argparse
import iris
import torch

def torch_dtype_from_str(datatype: str) -> torch.dtype:
    """Duplicated in many files"""
    dtype_map = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map.get(datatype, torch.float16)

def parse_args():
    """Duplicated argument parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--datatype", default="fp16")
    parser.add_argument("-w", "--num_warmup", type=int, default=1)
    parser.add_argument("-n", "--num_experiments", type=int, default=10)
    # ... more arguments
    return vars(parser.parse_args())

def bench_load(shmem, source_buffer, result_buffer, dtype,
               num_experiments=10, num_warmup=1):
    """Manual timing and statistics"""
    cur_rank = shmem.get_rank()
    n_elements = source_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    def run_kernel():
        if cur_rank == 0:
            load_kernel[grid](source_buffer, result_buffer, n_elements)
    
    # Manual warmup
    for _ in range(num_warmup):
        run_kernel()
        shmem.barrier()
    
    # Manual timing
    triton_ms = iris.do_bench(run_kernel, shmem.barrier,
                              n_repeat=num_experiments,
                              n_warmup=0)  # Already warmed up
    
    # Manual bandwidth calculation
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    total_bytes = n_elements * element_size_bytes
    bandwidth_gbps = total_bytes / (triton_ms * 1e-3) / 2**30
    
    print(f"Time: {triton_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth_gbps:.4f} GiB/s")
    
    return bandwidth_gbps

# Main
args = parse_args()
shmem = iris.iris(args["heap_size"])
dtype = torch_dtype_from_str(args["datatype"])
source_buffer = shmem.ones(args["buffer_size"], dtype=dtype)
result_buffer = shmem.zeros_like(source_buffer)

bandwidth = bench_load(shmem, source_buffer, result_buffer, dtype,
                       num_experiments=args["num_experiments"],
                       num_warmup=args["num_warmup"])
```

**Issues:**
- ~100+ lines of boilerplate
- Duplicated utility functions across 10+ files
- No standardized statistics (p50, p99)
- Manual warmup and timing
- No JSON export

## After: Using iris.bench

Clean, focused code:

```python
import torch
from iris.bench import benchmark, torch_dtype_from_str, compute_bandwidth_gbps

@benchmark(name="load_operation", warmup=5, iters=50, heap_size=1<<33)
def bench_load(shmem, buffer_size=1<<32, dtype_str="fp16"):
    """Clean benchmark using iris.bench"""
    # shmem is automatically created by the decorator
    
    dtype = torch_dtype_from_str(dtype_str)
    
    @setup
    def allocate_buffers():
        # Runs once before timing
        source_buffer = shmem.ones(buffer_size, dtype=dtype)
        result_buffer = shmem.zeros(buffer_size, dtype=dtype)
        return source_buffer, result_buffer
    
    @preamble
    def reset_output(source_buffer, result_buffer):
        # Runs before each timed iteration
        result_buffer.zero_()
    
    @measure
    def run_kernel(source_buffer, result_buffer):
        # This gets timed
        n_elements = source_buffer.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        load_kernel[grid](source_buffer, result_buffer, n_elements)

# Run benchmark
result = bench_load(buffer_size=1<<32, dtype_str="fp16")

# Automatic statistics available
result.print_summary()  # Shows mean, p50, p99, etc.

# Compute bandwidth using helper
element_size = torch.tensor([], dtype=torch_dtype_from_str("fp16")).element_size()
bandwidth = compute_bandwidth_gbps(1<<32 * element_size, result.mean_ms)
print(f"Bandwidth: {bandwidth:.2f} GiB/s")

# Export to JSON
result.to_json("results.json")
```

**Benefits:**
- ~50% less code (50 lines vs 100 lines)
- No duplicated utility functions
- Automatic statistics (mean, median, p50, p99)
- No manual warmup/timing logic
- JSON export included
- Cleaner code organization with @setup/@preamble/@measure

## Code Size Comparison

| Component | Before (lines) | After (lines) | Reduction |
|-----------|----------------|---------------|-----------|
| Utility functions | 15 | 1 (import) | 93% |
| Argument parsing | 25 | 0 (use params) | 100% |
| iris setup | 5 | 0 (automatic) | 100% |
| Warmup/timing | 15 | 0 (automatic) | 100% |
| Statistics | 5 | 0 (automatic) | 100% |
| Result output | 10 | 1 (print_summary) | 90% |
| **Total** | **~100** | **~50** | **~50%** |

## Migration Steps

1. **Replace manual setup with @benchmark decorator**
   - Remove manual `iris.iris()` creation
   - Add `shmem` as first parameter
   - Add @benchmark decorator with config

2. **Organize code with annotations**
   - Move tensor allocation to @setup
   - Move per-iteration setup to @preamble
   - Mark kernel launch with @measure

3. **Remove boilerplate**
   - Delete duplicated utility functions (use `iris.bench.torch_dtype_from_str`)
   - Remove manual warmup loops
   - Remove manual timing code
   - Remove manual statistics computation

4. **Use structured output**
   - Replace manual printing with `result.print_summary()`
   - Use `result.to_json()` for CI integration

## Parameter Sweeps

### Before
```python
for size in [1024, 2048, 4096]:
    for dtype_str in ["fp16", "fp32"]:
        result = bench_func(size, dtype_str)
        # Manual result tracking
        results.append({"size": size, "dtype": dtype_str, "time": result})
```

### After
```python
results = []
for size in [1024, 2048, 4096]:
    for dtype_str in ["fp16", "fp32"]:
        result = bench_func(size=size, dtype_str=dtype_str)
        results.append(result.to_dict())

# Export all results
import json
with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Best Practices

1. **Use @setup for expensive one-time operations**
   - Tensor allocation
   - Data initialization
   - Configuration setup

2. **Use @preamble for state reset**
   - Zeroing output buffers
   - Resetting flags
   - Clearing caches

3. **Keep @measure focused**
   - Only the kernel launch
   - The operation being benchmarked
   - No setup or teardown code

4. **Leverage automatic features**
   - Let decorator handle iris instance creation
   - Use automatic barrier synchronization
   - Trust automatic statistics computation

## Examples

See `examples/benchmark/bench_harness_example.py` for complete working examples.

## License

MIT License - Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
