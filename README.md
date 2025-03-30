# Iris

Iris is a Python- and Triton-based library that provide SHMEM-like RDMA support in Triton.

Iris matches PyTorch APIs on the host side:
```python
import iris

heap_size = 2**30
shmem = iris.Iris(heap_size)
buffer_size = 4096
buffer = shmem.zeros(buffer_size, device="cuda", dtype=torch.float32)
```

And matches Triton APIs on the device side:
```python
import iris

@triton.jit
def producer_kernel(buffer, heap_bases_ptr):
    source_rank = 0
    target_rank = 1
    values = iris.get(buffer, source_rank, target_rank, heap_bases_ptr)
```
## Examples

1. [P2P](./examples/p2p/README.md)
2. [Stream-K + Iris](./examples/stream-k/README.md)

## Getting started

We provide both a Docker and Apptainer files that sets up all dependencies.
### Docker
To build the image:

```shell
cd ./docker
./build.sh
```

And then to run the image,
```shell
./docker/run.sh
```

### Apptainer
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

## Installation

To install the package normally:

```shell
pip install .
```


To install in editable mode (auto-reloads on code changes):

```shell
pip install -e .
```