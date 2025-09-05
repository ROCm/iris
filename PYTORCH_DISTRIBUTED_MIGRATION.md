# PyTorch Distributed Migration Guide

This document describes the migration from MPI to PyTorch distributed in Iris.

## Overview

Iris has been updated to use PyTorch distributed instead of MPI for multi-GPU communication. This change:

- Removes the `mpi4py` dependency
- Replaces `mpirun` with PyTorch's `torch.distributed.elastic.multiprocessing`
- Maintains the same Iris API for existing applications
- Provides better integration with PyTorch ecosystem

## Key Changes

### Dependencies
- **Removed**: `mpi4py` 
- **Added**: Uses `torch.distributed` (included with PyTorch)

### Initialization Pattern

**Before (MPI):**
```bash
mpirun -np 8 python my_iris_app.py
```

**After (PyTorch Distributed):**
```python
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes

def _worker(local_rank: int, world_size: int, init_url: str, heap_size_bytes: int):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank
    )

    # Your Iris code here - same as before!
    import iris
    iris_ctx = iris.iris(heap_size_bytes)
    
    dist.barrier()
    dist.destroy_process_group()

def main(nprocs: int = 2, heap_size_bytes: int = 1 << 20):
    init_url = "tcp://127.0.0.1:29500"
    start_processes(
        fn=_worker,
        args=(nprocs, init_url, heap_size_bytes),
        nprocs=nprocs,
        join=True,
    )

if __name__ == "__main__":
    n = torch.cuda.device_count() or 2
    main(nprocs=n, heap_size_bytes=1 << 20)
```

### Existing Iris Code

**Good news**: Your existing Iris application code doesn't need to change! The Iris API remains the same:

```python
import iris

# This still works exactly the same
iris_ctx = iris.iris(heap_size=1 << 30)
tensor = iris_ctx.zeros(1000, 1000, dtype=torch.float32)
iris_ctx.barrier()
```

The only change is how you launch your application (PyTorch distributed instead of `mpirun`).

## Migration Steps

1. **Update launch mechanism**: Replace `mpirun` with PyTorch distributed launcher
2. **Remove MPI**: No need to install or import `mpi4py`
3. **Optional**: Use the new `iris.launch_iris()` helper function

### Helper Function

Iris now provides a convenience function for launching:

```python
import iris

# Launch with 4 processes and 1GB heap per process
iris.launch_iris(nprocs=4, heap_size_bytes=1 << 30)
```

## Backwards Compatibility

- All existing Iris APIs work unchanged
- All Triton device functions (`iris.load`, `iris.store`, etc.) work unchanged  
- All Iris memory management functions work unchanged
- Examples and tutorials need launcher updates but core logic remains the same

## Examples

See `examples/pytorch_distributed_example.py` for a complete working example of the new pattern.

## FAQ

**Q: Do I need to rewrite my Iris kernels?**
A: No! All Triton kernels using Iris functions work unchanged.

**Q: What about existing examples?**
A: The core Iris code in examples works unchanged. Only the launcher (replacing `mpirun`) needs updating.

**Q: Can I still use MPI?**
A: No, MPI support has been removed. Use PyTorch distributed instead.

**Q: What PyTorch backends are supported?**
A: NCCL (recommended for GPU) and Gloo (fallback) are supported.