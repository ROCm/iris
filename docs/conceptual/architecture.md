# Iris Architecture

This document covers the system design and internals of the Iris framework based on the actual implementation.

## Overview

Iris is a multi-GPU communication and memory management framework that provides high-performance distributed operations across AMD GPUs using HIP, MPI, and Triton kernels.

## System Architecture

### 1. Application Layer
- High-level Python APIs through the `Iris` class
- PyTorch tensor integration
- User-facing interfaces for memory and communication operations

### 2. Framework Layer
- **Memory Management**: Symmetric heap allocation across GPUs
- **Communication Primitives**: Load/store, get/put, atomic operations
- **Synchronization**: MPI barriers and coordination

### 3. Runtime Layer
- **HIP Runtime**: AMD GPU compute and memory management
- **MPI Coordination**: Multi-node communication and rank management
- **Device Management**: GPU device selection and IPC handle management

### 4. Hardware Layer
- **AMD GPU Hardware**: MI300X, MI350X, MI355X, and other ROCm-compatible GPUs
- **Network Interfaces**: High-speed interconnects for GPU-to-GPU communication
- **Memory Hierarchy**: GPU memory, system memory, and shared memory

## Core Components

### Memory Management

The `Iris` class manages a symmetric heap across all GPUs:

```python
class Iris:
    def __init__(self, heap_size=1 << 30):
        # Initialize MPI communicator and rank information
        # Allocate symmetric heap on each GPU
        # Set up IPC handles for cross-GPU memory access
        # Establish heap base addresses for all ranks
```

**Key Features:**
- **Symmetric Heap**: Same memory layout across all GPUs
- **IPC Handles**: Cross-GPU memory sharing using AMD's IPC mechanism
- **Memory Pool**: Pre-allocated memory pool for efficient allocation
- **Alignment**: 1024-byte alignment for optimal performance

### Communication Engine

Iris provides several communication primitives:

#### Memory Operations
- **`load(pointer, to_rank, from_rank, heap_bases, mask=None)`**: Load data from remote GPU
- **`store(pointer, value, from_rank, to_rank, heap_bases, mask=None)`**: Store data to remote GPU
- **`get(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None)`**: Get data from remote GPU
- **`put(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None)`**: Put data to remote GPU

#### Atomic Operations
- **`atomic_add(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_sub(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_cas(pointer, cmp, val, from_rank, to_rank, heap_bases, sem=None, scope=None)`**
- **`atomic_xchg(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_xor(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_and(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_or(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_min(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**
- **`atomic_max(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None)`**

#### Tensor Creation
- **`zeros(*size, dtype=torch.int, device=None, requires_grad=False, **kwargs)`**
- **`ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`**
- **`randn(*size, dtype=torch.float, device=None, requires_grad=False)`**
- **`arange(start=0, end, step=1, dtype=None, layout=torch.strided, device=None, requires_grad=False)`**
- **`linspace(start, end, steps, dtype=torch.float)`**
- **`uniform(size, low=0.0, high=1.0, dtype=torch.float)`**
- **`randint(size, low, high, dtype=torch.int)`**

### Synchronization

- **MPI Barriers**: `world_barrier()` for global synchronization
- **Rank Coordination**: Automatic rank assignment and GPU mapping
- **Device Barriers**: `barrier()` method for device-level synchronization

## Design Principles

### 1. Simplicity
- **Minimal API Surface**: Single `Iris` class with intuitive methods
- **Clear Semantics**: Each operation has well-defined behavior
- **Consistent Patterns**: Similar parameter ordering across operations

### 2. Performance
- **Zero-Copy Operations**: Direct memory access through IPC handles
- **Efficient Memory Layouts**: Symmetric heap for optimal access patterns
- **Optimized Communication**: HIP-native operations for maximum throughput

### 3. Flexibility
- **Custom Communication Patterns**: Build complex patterns from primitive operations
- **Extensible Primitives**: Easy to add new atomic operations
- **Platform Independence**: Works across different AMD GPU generations

## Implementation Details

### Memory Layout
- **Contiguous Allocation**: Single large memory pool per GPU
- **Proper Alignment**: 1024-byte alignment for optimal performance
- **NUMA Awareness**: Respects GPU memory topology

### Communication Protocols
- **IPC Handles**: AMD's inter-process communication for cross-GPU access
- **Error Handling**: Robust error checking and validation
- **Recovery Mechanisms**: Graceful handling of communication failures

### Triton Integration
- **Kernel Generation**: Automatic kernel generation for complex operations
- **Memory Access Patterns**: Optimized memory access through Triton
- **Performance Tuning**: Block size and grid optimization

## Performance Characteristics

### Bandwidth
- **Local Memory**: ~5+ TB/s on MI300X
- **Remote Memory**: ~47 GB/s between GPUs
- **Atomic Operations**: ~15 GB/s for remote atomic operations

### Latency
- **Local Operations**: Sub-microsecond
- **Remote Operations**: Microsecond to millisecond depending on operation type
- **Barrier Synchronization**: Microsecond level for small numbers of GPUs

## Scalability

### Multi-GPU Support
- **Current**: Tested up to 8 GPUs (MI300X)
- **Theoretical**: Limited by MPI implementation and hardware
- **Memory**: Linear scaling with number of GPUs

### Multi-Node Support
- **MPI Integration**: Works with any MPI implementation
- **Network Topology**: Performance depends on interconnect bandwidth
- **Load Balancing**: Automatic rank-to-GPU mapping

---

This architecture provides a solid foundation for high-performance multi-GPU applications while maintaining simplicity and flexibility for developers.
