---
myst:
  html_meta:
    "description": "Learn about Iris, AMD's powerful multi-GPU programming framework built on Triton"
    "keywords": "Iris, AMD, GPU, Multi-GPU, Triton, SHMEM, RMA, Distributed Computing"
---

# Iris Documentation

Welcome to the Iris documentation! Iris is AMD's powerful and intuitive multi-GPU programming framework built on Triton, designed to make distributed GPU computing accessible to everyone.

## What is Iris?

Iris is a **Triton-based framework for Remote Memory Access (RMA)** that provides SHMEM-like APIs for multi-GPU programming. It's designed by experts and built for scale, ensuring you get a robust and efficient experience.

### Key Features

- **🚀 Simple & Intuitive**: Write multi-GPU programs without the complexity
- **⚡ High Performance**: Inherits Triton's programmability and performance  
- **🔧 Familiar APIs**: SHMEM-like patterns, Triton-style device APIs, and PyTorch-like host APIs
- **🏗️ Expert Design**: Built from scratch by GPU and distributed computing experts
- **⚡ Minimal Dependencies**: Only Triton, PyTorch, HIP runtime, and mpi4py

## Documentation Structure

This documentation follows the [Diataxis](https://diataxis.fr/) framework to provide effective learning materials:

### 📚 **Getting Started**
- **Installation**: Set up Iris on your system
- **Quick Start**: Run your first multi-GPU program
- **Setup Alternatives**: Different ways to get Iris running

### 🎯 **Tutorials** 
- **Basic Operations**: Load/store operations between GPUs
- **Atomic Operations**: Cross-GPU atomic operations
- **Message Passing**: Point-to-point communication patterns
- **GEMM Examples**: Matrix multiplication with communication

### 🔧 **How-to Guides**
- **Performance Optimization**: Tips and tricks for best performance
- **Debugging**: Common issues and solutions
- **Benchmarking**: Measure and analyze performance

### 🧠 **Conceptual**
- **Programming Model**: Deep dive into how Iris works
- **Fine-grained Overlap**: Advanced optimization techniques
- **Architecture**: System design and internals

### 📖 **Reference**
- **API Reference**: Complete API documentation
- **Examples**: Comprehensive example collection
- **Contributing**: How to contribute to Iris

## Quick Start

```bash
# Install Iris
pip install iris

# Run your first example
mpirun -np 8 python examples/00_load/load_bench.py
```

## Supported GPUs

Iris currently supports:
- MI300X, MI350X & MI355X
- Other AMD GPUs with ROCm compatibility

## Community & Support

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Contributing**: Help make Iris better for everyone

---

**Ready to start your multi-GPU journey? Let's begin with the [Installation Guide](getting-started/installation.md)!**
