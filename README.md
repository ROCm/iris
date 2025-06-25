<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Iris: First-Class Multi-GPU Programming Experience in Triton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/ROCm/iris/blob/main/.github/workflows/lint.yml)

> [!IMPORTANT]  
> This project is intended for research purposes only and is provided by AMD Research and Advanced Development team.  This is not a product. Use it at your own risk and discretion.


Iris is a Triton-based framework for Remote Memory Access (RMA) operations. Iris provides SHMEM-like APIs within Triton for Multi-GPU programming. Iris' goal is to make Multi-GPU programming a first-class citizen in Triton while retaining Triton's programmability and performance.

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

## Supported GPUs

Iris currently supports:

- MI300X

**Note**: 
> [!NOTE]
> Iris may work on other AMD GPUs with ROCm compatibility, but has only been tested on MI300X.

## Roadmap

We plan to extend Iris with the following features:

- **Extended GPU Support**: Testing and optimization for other AMD GPUs beyond MI300X
- **RDMA Support**: Multi-node support using Remote Direct Memory Access (RDMA) for distributed computing across multiple machines
- **More Code Examples**: Comprehensive examples covering various use cases and patterns

# Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to set up your development environment and contribute to the project.



## Support

Need help? We're here to support you! Here are a few ways to get in touch:

1. **Open an Issue**: Found a bug or have a feature request? [Open an issue](https://github.com/ROCm/iris/issues/new/choose) on GitHub
2. **Contact the Team**: If GitHub issues aren't working for you or you need to reach us directly, feel free to contact our development team

We welcome your feedback and contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
