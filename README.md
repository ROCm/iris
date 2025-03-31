# Iris

Iris is a Python- and Triton-based library that provide SHMEM-like RDMA support in Triton.

## Key Features

- **SHMEM-like RMA**: Iris provides SHMEM-like RMA support in Triton.
- **Simple and Intuitive API**: Iris provides simple and intuitive RMA APIs. Writing multi-GPU programs is as easy as writing single-GPU programs.
- **Triton-based**: Iris is built on top of Triton and inherits Triton's performance and capabilities.

## Examples

Iris matches PyTorch APIs on the host side and Triton APIs on the device side:
```python
import torch
import triton.language as tl

import iris

@triton.jit
def kernel(buffer, buffer_size, block_size, heap_bases_ptr):

    # Compute start index of this block
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    
    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Store 1 in the target buffer at each offset
    source_rank = 0
    target_rank = 1
    iris.put(buffer + offsets, 1,
            source_rank, target_rank,
            heap_bases_ptr, mask=mask)

heap_size = 2**30
buffer_size = 4096
block_size = 1024
shmem = iris.Iris(heap_size)
cur_rank = shmem.get_rank()
buffer = shmem.zeros(buffer_size, device="cuda", dtype=torch.float32)
grid = lambda meta: (triton.cdiv(buffer_size, meta["BLOCK_SIZE"]),)

source_rank = 0
if cur_rank == source_rank:
    put_kernel[grid](
        buffer,
        buffer_size,
        block_size,
        shmem.get_heap_bases(),
    )
shmem.barrier()    
```

## Examples

1. [P2P](./examples/p2p/README.md)
2. [Stream-K + Iris](./examples/stream-k/README.md)


## Installation

To install the package normally:

```shell
pip install .
```


To install in editable mode (auto-reloads on code changes):

```shell
pip install -e .
```

## Getting started

We provide both a Docker and Apptainer files that sets up all dependencies.
#### Docker
To build the image:

```shell
cd ./docker
./build.sh
```

And then to run the image,
```shell
./docker/run.sh
```

#### Apptainer
To build the image:
```shell
./apptainer/build.sh
```

And then to run the image,
```shell
./apptainer/run.sh
```

Once inside the Apptainer image, source the `activate.sh` script.

```
source activate.sh
```

