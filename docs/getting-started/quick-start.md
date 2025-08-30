# Quick Start Guide

This guide will help you run your first multi-GPU program with Iris.

## Overview

In this quick start, you'll learn how to:
1. Initialize Iris
2. Create a simple multi-GPU program
3. Run basic communication operations

## Prerequisites

- Iris installed (see [Installation Guide](installation.md))
- At least 2 AMD GPUs available
- MPI environment configured

## Your First Iris Program

Create a file called `hello_iris.py`:

```python
import iris
import torch

def main():
    # Initialize Iris
    iris.init()

    # Get rank and size information
    rank = iris.rank()
    size = iris.size()

    print(f"Hello from rank {rank} of {size}")

    # Create a simple tensor
    local_tensor = torch.tensor([rank], device='cuda')

    # All-gather operation
    gathered = iris.all_gather(local_tensor)

    print(f"Rank {rank}: gathered = {gathered}")

    # Synchronize all ranks
    iris.barrier()

    # Finalize Iris
    iris.finalize()

if __name__ == "__main__":
    main()
```

## Running the Program

```bash
# Run with 4 GPUs
mpirun -np 4 python hello_iris.py

# Expected output:
# Hello from rank 0 of 4
# Hello from rank 1 of 4
# Hello from rank 2 of 4
# Hello from rank 3 of 4
# Rank 0: gathered = tensor([0, 1, 2, 3], device='cuda:0')
# Rank 1: gathered = tensor([0, 1, 2, 3], device='cuda:1')
# Rank 2: gathered = tensor([0, 1, 2, 3], device='cuda:2')
# Rank 3: gathered = tensor([0, 1, 2, 3], device='cuda:3')
```

## Understanding the Code

### Initialization
```python
iris.init()  # Initialize Iris framework
```

### Rank Information
```python
rank = iris.rank()  # Current GPU rank (0, 1, 2, ...)
size = iris.size()  # Total number of GPUs
```

### Communication
```python
gathered = iris.all_gather(local_tensor)  # Collect data from all GPUs
```

### Synchronization
```python
iris.barrier()  # Wait for all GPUs to reach this point
```

### Cleanup
```python
iris.finalize()  # Clean up Iris resources
```

## Next Steps

Now that you've run your first program:

1. **Explore Examples**: Check out the examples directory for more complex patterns
2. **Learn Operations**: Read about basic operations
3. **Understand Concepts**: Dive into the programming model

## Troubleshooting

If you encounter issues:

- Check that all GPUs are visible: `nvidia-smi` or `rocm-smi`
- Verify MPI installation: `mpirun --version`
- Ensure Iris is properly installed: `python -c "import iris"`

---

*This is a placeholder document. Full content will be added in future updates.*
