# Installation Guide

This guide covers how to install Iris on your system.

## Overview

Iris requires several dependencies including Python, PyTorch, HIP runtime, and MPI. This guide will walk you through the installation process.

## Prerequisites

### System Requirements

- Linux operating system (Ubuntu 20.04+ recommended)
- AMD GPU with ROCm support
- Python 3.8+
- CUDA toolkit (for PyTorch compatibility)

### Required Software

- Python 3.8+
- PyTorch 2.0+
- ROCm HIP runtime
- MPI implementation (OpenMPI or MPICH)
- Git

## Installation Methods

### 1. Using pip (Recommended)

```bash
# Install from PyPI
pip install iris

# Or install from source
pip install git+https://github.com/amd/iris.git
```

### 2. From Source

```bash
# Clone the repository
git clone https://github.com/amd/iris.git
cd iris

# Install in development mode
pip install -e .
```

### 3. Using Conda

```bash
# Create conda environment
conda create -n iris python=3.9
conda activate iris

# Install dependencies
conda install pytorch torchvision torchaudio -c pytorch
pip install iris
```

## Verification

After installation, verify that Iris is working:

```python
import iris
print(f"Iris version: {iris.__version__}")

# Test basic functionality
iris.init()
print(f"Rank: {iris.rank()}, Size: {iris.size()}")
iris.finalize()
```

## Troubleshooting

### Common Issues

1. **HIP runtime not found**: Ensure ROCm is properly installed
2. **MPI not found**: Install OpenMPI or MPICH
3. **PyTorch compatibility**: Use compatible PyTorch version

### Getting Help

- Check the [debugging guide](how-to/debug-common-issues.md)
- Open an issue on GitHub
- Join community discussions

---

*This is a placeholder document. Full content will be added in future updates.*
