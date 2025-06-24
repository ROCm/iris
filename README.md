<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Iris: First-Class Multi-GPU Programming Experience in Triton

Iris is a Triton-based framework for Remote Direct Memory Access (RDMA) operations. Iris provides SHMEM-like APIs within Triton for Multi-GPU programming. Iris' goal is to make Multi-GPU programming a first-class citizen in Triton while retaining Triton’s programmability and performance.

## Key Features

- **SHMEM-like RMA**: Iris provides SHMEM-like RMA support in Triton.
- **Simple and Intuitive API**: Iris provides simple and intuitive RMA APIs. Writing multi-GPU programs is as easy as writing single-GPU programs.
- **Triton-based**: Iris is built on top of Triton and inherits Triton's performance and capabilities.

## Documentation

1. [Peer-to-Peer Communication](examples/README.md)
2. [Fine-grained GEMM & Communication Overlap](./docs/FINEGRAINED_OVERLAP.md)

## API Example

Iris matches PyTorch APIs on the host side and Triton APIs on the device side:
```python
import torch
import triton
import triton.language as tl
import iris

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

heap_size = 2**30
buffer_size = 4096
block_size = 1024
shmem = iris.Iris(heap_size)
cur_rank = shmem.get_rank()
buffer = shmem.zeros(buffer_size, device="cuda", dtype=torch.float32)
grid = lambda meta: (triton.cdiv(buffer_size, meta["block_size"]),)

source_rank = 0
if cur_rank == source_rank:
    kernel[grid](
        buffer,
        buffer_size,
        block_size,
        shmem.get_heap_bases(),
    )
shmem.barrier() 
```

## Quick Start Guide (using Docker)

Using docker compose, you can get started with a simple dev environment where the active Iris directory is mounted inside the docker container. This way, any changes you make outside the container to Iris are reflected inside the container (getting set up with a vscode instance becomes easy!)

```shell
docker compose up --build -d
docker attach iris-dev
cd iris && pip install -e .
```

## Getting started

### Docker

```shell
./docker/build.sh <image-name>
./docker/run.sh <image-name>
cd iris && pip install -e .
```

### Apptainer
```shell
./apptainer/build.sh
./apptainer/run.sh
source activate.sh
```
