<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

<p align="center">
  <img src="docs/images/logo.png" width="300px" />
</p>

# Iris: First-Class Multi-GPU Programming Experience in Triton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/ROCm/iris/blob/main/.github/workflows/lint.yml) [![Iris Tests](https://github.com/ROCm/iris/actions/workflows/iris-tests-apptainer.yml/badge.svg)](https://github.com/ROCm/iris/actions/workflows/iris-tests-apptainer.yml)

Iris is a Triton-based framework for Remote Memory Access (RMA) operations developed by AMD's Research and Advanced Development team. Iris provides SHMEM-like APIs within Triton for Multi-GPU programming. Iris' goal is to make Multi-GPU programming a first-class citizen in Triton while retaining Triton's programmability and performance.

## Key Features

- **SHMEM-like RMA**: Iris provides SHMEM-like RMA support in Triton.
- **Simple and Intuitive API**: Iris provides simple and intuitive RMA APIs. Writing multi-GPU programs is as easy as writing single-GPU programs.
- **Triton-based**: Iris is built on top of Triton and inherits Triton's performance and capabilities.
- **Gluon-style Aggregate API (Experimental)**: Optional cleaner API using Triton's `@aggregate` decorator for better encapsulation.

## Documentation

- [API Reference](https://rocm.github.io/iris/reference/api-reference.html)
- [Programming Model](https://rocm.github.io/iris/conceptual/programming-model.html)
- [Examples](https://rocm.github.io/iris/reference/examples.html)
- [Fine-grained GEMM & Communication Overlap](https://rocm.github.io/iris/conceptual/finegrained-overlap.html)
- [Setup Alternatives](https://rocm.github.io/iris/getting-started/installation.html)

## API Example

Here's a simple example showing how to perform remote memory operations between GPUs using Iris:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import iris

# Device-side APIs
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

def _worker(rank, world_size):
    # Torch distributed initialization
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )

    # Iris initialization
    heap_size = 2**30   # 1GiB symmetric heap for inter-GPU communication
    iris_ctx = iris.iris(heap_size)
    cur_rank = iris_ctx.get_rank()

    # Iris tensor allocation
    buffer_size = 4096  # 4K elements buffer
    buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)

    # Launch the kernel on rank 0
    block_size = 1024
    grid = lambda meta: (triton.cdiv(buffer_size, meta["block_size"]),)
    source_rank = 0
    if cur_rank == source_rank:
        kernel[grid](
            buffer,
            buffer_size,
            block_size,
            iris_ctx.get_heap_bases(),
        )

    # Synchronize all ranks
    iris_ctx.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Using two ranks
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
```

### Alternative: Gluon-style Aggregate API (Experimental)

Iris also provides an experimental cleaner API using Triton's Gluon with `@gluon.jit` decorator:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl

# Device-side APIs - context encapsulates heap_bases
@gluon.jit
def kernel(IrisDeviceCtx: gl.constexpr, context_tensor,
          buffer, buffer_size: gl.constexpr, block_size: gl.constexpr):
    # Initialize device context from tensor
    ctx = IrisDeviceCtx.initialize(context_tensor)
    
    pid = gl.program_id(0)
    block_start = pid * block_size
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, block_size, layout=layout)
    mask = offsets < buffer_size

    # Store 1 in the target buffer - no need to pass heap_bases separately!
    target_rank = 1
    ctx.store(buffer + offsets, 1, target_rank, mask=mask)

def _worker(rank, world_size):
    # Torch distributed initialization
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )

    # Iris initialization
    heap_size = 2**30   # 1GiB symmetric heap
    iris_ctx = iris_gl.iris(heap_size)
    context_tensor = iris_ctx.get_device_context()  # Get encoded context
    cur_rank = iris_ctx.get_rank()
    
    # Iris tensor allocation
    buffer_size = 4096  # 4K elements buffer
    buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)
    
    # Launch the kernel on rank 0
    block_size = 1024
    grid = (buffer_size + block_size - 1) // block_size
    source_rank = 0
    if cur_rank == source_rank:
        kernel[(grid,)](iris_gl.IrisDeviceCtx, context_tensor, 
                       buffer, buffer_size, block_size, num_warps=1)

    # Synchronize all ranks
    iris_ctx.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Using two ranks
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
```

## Quick Start Guide

### Quick Installation

> [!NOTE]
> **Requirements**: Python 3.10+, PyTorch 2.0+ (ROCm version), ROCm 6.3.1+ HIP runtime, Triton, and setuptools>=61

For a quick installation directly from the repository:

```shell
pip install git+https://github.com/ROCm/iris.git
```

### Docker Compose (Recommended for Development)

The recommended way to get started is using Docker Compose, which provides a development environment with the Iris directory mounted inside the container. This allows you to make changes to the code outside the container and see them reflected inside.

```shell
# Start the development container
docker compose up --build -d

# or depending on your docker version
docker-compose up --build -d

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode
cd iris && pip install -e .
```

For baremetal install, Docker or Apptainer setup, see [Installation](https://rocm.github.io/iris/getting-started/installation.html).

## Next Steps

Check out our [examples](examples/) directory for ready-to-run scripts and usage patterns, including peer-to-peer communication and GEMM benchmarks.

## Supported GPUs

Iris currently supports:

- MI300X, MI350X & MI355X

> [!NOTE]
> Iris may work on other AMD GPUs with ROCm compatibility.

## Roadmap

We plan to extend Iris with the following features:

- **Extended GPU Support**: Testing and optimization for other AMD GPUs.
- **RDMA Support**: Multi-node support using Remote Direct Memory Access (RDMA) for distributed computing across multiple machines.
- **End-to-End Integration**: Comprehensive examples covering various use cases and end-to-end patterns.

# Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to set up your development environment and contribute to the project.

## Support

Need help? We're here to support you! Here are a few ways to get in touch:

1. **Open an Issue**: Found a bug or have a feature request? [Open an issue](https://github.com/ROCm/iris/issues/new/choose) on GitHub
2. **Contact the Team**: If GitHub issues aren't working for you or you need to reach us directly, feel free to contact our development team

We welcome your feedback and contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
