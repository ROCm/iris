# GEMM Examples Tutorial

This tutorial covers matrix multiplication examples with communication in Iris.

## Overview

GEMM (General Matrix Multiplication) is a fundamental operation that often requires communication between GPUs. Iris provides efficient primitives for implementing distributed GEMM operations.

## Basic Distributed GEMM

```python
import iris
import torch

# Initialize Iris
iris.init()

# Matrix dimensions
M, N, K = 4096, 4096, 4096

# Create local matrices
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')
C = torch.zeros(M, N, device='cuda')

# Perform local GEMM
C = torch.mm(A, B)

# Communication and reduction
# Implementation details...
```

## Communication Patterns

### All-Reduce Pattern

```python
# All-reduce pattern for distributed GEMM
# Each GPU computes partial result, then reduces across all GPUs

# Local computation
local_C = torch.mm(A, B)

# All-reduce to get final result
iris.all_reduce(local_C, op='sum')
```

### All-Scatter Pattern

```python
# All-scatter pattern for distributed GEMM
# Distribute work across GPUs, then gather results

# Scatter input matrices
# Implementation details...
```

## Performance Optimization

- Use appropriate tile sizes
- Overlap computation and communication
- Profile memory access patterns

---

*This is a placeholder document. Full content will be added in future updates.*
