# Quick Start Guide

This guide will help you run your first multi-GPU program with Iris.

## Overview

In this quick start, you'll learn how to:
1. Set up the Iris development environment
2. Initialize Iris with proper context
3. Create a simple multi-GPU program using Triton kernels
4. Run basic communication operations

## Prerequisites

- Docker and Docker Compose installed
- At least 2 AMD GPUs available (or use the provided container environment)
- Basic understanding of Triton and PyTorch

## Setup

The recommended way to get started is using Docker Compose:

```shell
# Clone the repository
git clone https://github.com/ROCm/iris.git
cd iris

# Start the development container
docker compose up --build -d

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode
pip install -e .
```

For manual Docker or Apptainer setup, see the [Installation Guide](installation.md).

## Your First Iris Program

Create a file called `hello_iris.py`:

```python
import torch
import triton
import triton.language as tl
import iris

# Device-side kernel using Triton
@triton.jit
def kernel(buffer, buffer_size: tl.constexpr, block_size: tl.constexpr, heap_bases_ptr):
    # Compute start index of this block
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Store 1 in the target buffer at each offset
    source_rank = 0
    target_rank = 1
    iris.store(buffer + offsets, 1,
            source_rank, target_rank,
            heap_bases_ptr, mask=mask)

def main():
    # Initialize Iris with symmetric heap
    heap_size = 2**30   # 1GiB symmetric heap for inter-GPU communication
    iris_ctx = iris.iris(heap_size)
    cur_rank = iris_ctx.get_rank()
    world_size = iris_ctx.get_num_ranks()

    print(f"Hello from rank {cur_rank} of {world_size}")

    # Iris tensor allocation
    buffer_size = 4096  # 4K elements buffer
    buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)

    # Launch the kernel on rank 0
    block_size = 1024
    grid = lambda meta: (triton.cdiv(buffer_size, meta["block_size"]),)
    source_rank = 0

    if cur_rank == source_rank:
        print(f"Rank {cur_rank}: Launching kernel to store data to rank 1")
        kernel[grid](
            buffer,
            buffer_size,
            block_size,
            iris_ctx.get_heap_bases(),
        )

    # Synchronize all ranks
    iris_ctx.barrier()

    if cur_rank == 1:
        print(f"Rank {cur_rank}: Data received from rank 0")
        # Verify the data was stored
        data_sum = buffer.sum().item()
        print(f"Rank {cur_rank}: Sum of received data = {data_sum}")

if __name__ == "__main__":
    main()
```

## Running the Program

```bash
# Run with 2 GPUs
mpirun -np 2 python hello_iris.py

# Expected output:
# Hello from rank 0 of 2
# Hello from rank 1 of 2
# Rank 0: Launching kernel to store data to rank 1
# Rank 1: Data received from rank 0
# Rank 1: Sum of received data = 4096.0
```

## Understanding the Code

### Iris Context Initialization
```python
iris_ctx = iris.iris(heap_size)  # Create Iris context with symmetric heap
```

### Rank Information
```python
cur_rank = iris_ctx.get_rank()      # Current GPU rank (0, 1, 2, ...)
world_size = iris_ctx.get_num_ranks()  # Total number of GPUs
```

### Tensor Allocation
```python
buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)
```

### Remote Memory Operations
```python
iris.store(buffer + offsets, 1, source_rank, target_rank, heap_bases_ptr, mask=mask)
```

### Synchronization
```python
iris_ctx.barrier()  # Wait for all GPUs to reach this point
```

The Iris context will be automatically cleaned up when the program ends.

## Try the Examples

Once you have Iris installed, you can run the provided examples:

### Basic Operations
```bash
# Load operations across multiple GPUs
mpirun -np 8 python examples/00_load/load_bench.py

# Store operations across multiple GPUs
mpirun -np 8 python examples/01_store/store_bench.py

# Atomic operations
mpirun -np 8 python examples/04_atomic_add/atomic_add_bench.py
```

### GEMM Operations
```bash
# Matrix multiplication with all-scatter communication
mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py --benchmark --validate

# Matrix multiplication with all-reduce using atomics
mpirun -np 8 python examples/08_gemm_atomics_all_reduce/benchmark.py --benchmark --validate
```

## Next Steps

Now that you've run your first program:

1. **Explore Examples**: Check out the [examples directory](../../examples/) for more complex patterns
2. **Learn Operations**: Read about [basic operations](../conceptual/programming-model.md)
3. **Understand Concepts**: Dive into the [programming model](../conceptual/programming-model.md)
