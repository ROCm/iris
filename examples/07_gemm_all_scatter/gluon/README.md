# Gluon-based GEMM All-Scatter

This directory contains the Gluon port of the GEMM All-Scatter example, demonstrating how to use Iris with Gluon's `@gluon.jit` decorator and `gl.*` language primitives.

## Files

- **gemm_all_scatter.py**: Core GEMM kernel using `@gluon.jit` and `IrisDeviceCtx` aggregate
- **matmul_wrapper.py**: PyTorch autograd wrapper for the Gluon GEMM kernel
- **benchmark.py**: Benchmark script for the Gluon-based GEMM All-Scatter

## Key Differences from Traditional Iris

### Context Encoding
Instead of passing `heap_bases` directly, the Gluon version uses context encoding:

```python
# Host side
ctx = iris_gl.iris(heap_size=2**30)
context_tensor = ctx.get_device_context()  # [cur_rank, num_ranks, heap_bases...]

# Kernel launch
gemm_kernel[(num_sms,)](
    iris_gl.IrisDeviceCtx,  # Pass aggregate class
    context_tensor,         # Pass encoded context
    A, B, C, ...
)
```

### Device Side
```python
@gluon.jit
def kernel(IrisDeviceCtx: gl.constexpr, context_tensor, ...):
    # Initialize context
    ctx = IrisDeviceCtx.initialize(context_tensor)
    
    # Use gl.* primitives
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)
    a = gl.load(A_BASE)
    b = gl.load(B_BASE)
    acc += gl.dot(a, b)
    
    # Inter-rank communication
    ctx.store(c_global + offset, c, remote_rank, mask=mask)
```

## Usage

Run the benchmark with:

```bash
python benchmark.py -m 8192 -n 4608 -k 36864 --validate --benchmark -r 2
```

## Technical Notes

- Uses `gl.BlockedLayout([1], [64], [1], [0])` for `gl.arange()` operations (AMD GPUs)
- All GEMM operations use `gl.*` primitives: `gl.load`, `gl.store`, `gl.dot`, `gl.zeros`
- Context methods (`ctx.store()`, `ctx.load()`) handle inter-rank communication
- Maintains all optimizations from original example: persistent kernel, tiling, blocking, compiler hints
