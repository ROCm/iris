---
myst:
  html_meta:
    "description": "Iris: First-Class Multi-GPU Programming Experience in Triton"
    "keywords": "Iris, AMD, GPU, Multi-GPU, Triton, SHMEM, RMA, Distributed Computing"
---

# Iris

<div style="text-align: center; margin: 2rem 0;">
  <img src="/images/logo.png" alt="Iris Logo" class="only-dark" style="width: 400px; height: auto;">
  <p style="font-size: 1.2rem; color: #666; margin: 0;">First-Class Multi-GPU Programming Experience in Triton</p>
</div>

## What is Iris?

Iris is a **Triton-based framework for Remote Memory Access (RMA)** operations. Iris provides SHMEM-like APIs within Triton for Multi-GPU programming. Iris' goal is to make Multi-GPU programming a first-class citizen in Triton while retaining Triton's programmability and performance.

### Key Features

- **SHMEM-like RMA**: Iris provides SHMEM-like RMA support in Triton
- **Simple and Intuitive API**: Iris provides simple and intuitive RMA APIs. Writing multi-GPU programs is as easy as writing single-GPU programs
- **Triton-based**: Iris is built on top of Triton and inherits Triton's performance and capabilities

## Quick Start

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

### Run Your First Example

Here's a simple example showing how to perform remote memory operations between GPUs using Iris:

```python
import torch
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
```

For more examples, check out the [examples directory](../../examples/) with ready-to-run scripts and usage patterns.

For other setup methods, see the [Installation Guide](getting-started/installation.md).

## Documentation Structure

### 📚 **Getting Started**
- **[Installation](getting-started/installation.md)**: Set up Iris on your system
- **[Quick Start](getting-started/quick-start.md)**: Run your first multi-GPU program

### 🧠 **Conceptual**
- **[Programming Model](conceptual/programming-model.md)**: How Iris works
- **[Fine-grained Overlap](conceptual/finegrained-overlap.md)**: GEMM & communication overlap
- **[Architecture](conceptual/architecture.md)**: System design and internals

### 📖 **Reference**
- **[API Reference](reference/iris-api.md)**: Auto-generated API documentation
- **[Examples](reference/examples.md)**: Working code examples
- **[Contributing](reference/contributing.md)**: How to contribute

## Supported GPUs

Iris currently supports:
- MI300X, MI350X & MI355X

> **Note**: Iris may work on other AMD GPUs with ROCm compatibility.

## Roadmap

We plan to extend Iris with the following features:

- **Extended GPU Support**: Testing and optimization for other AMD GPUs
- **RDMA Support**: Multi-node support using Remote Direct Memory Access (RDMA) for distributed computing across multiple machines
- **End-to-End Integration**: Comprehensive examples covering various use cases and end-to-end patterns

## Community & Support

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Contributing**: Help make Iris better for everyone

---

**Ready to start your multi-GPU journey? Begin with the [Installation Guide](getting-started/installation.md)!**
