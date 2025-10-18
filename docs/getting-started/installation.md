# Installation Guide

This guide covers how to install Iris on your system using various methods.

## Overview

Iris has minimal dependencies including Python, PyTorch, ROCm HIP runtime, and Triton. This guide will walk you through the installation process using different approaches.

## Prerequisites

### System Requirements

- Linux operating system (Ubuntu 22.04+)
- AMD GPU with ROCm 6.3.1+ support (MI300X, MI350X, MI355X, or other ROCm-compatible GPUs)

### Required Software

**Minimum working requirements based on the Docker setup:**

- Python 3.10+
- PyTorch 2.0+ (ROCm version)
- ROCm 6.3.1+ HIP runtime
- Git
- Triton (suggested commit: [dd5823453bcc7973eabadb65f9d827c43281c434](https://github.com/triton-lang/triton/tree/dd5823453bcc7973eabadb65f9d827c43281c434))

**Note**: These versions represent the minimum working configuration. Using different versions may cause compatibility issues.

### Gluon Backend Requirements (Experimental)

To use the experimental Gluon APIs, additional requirements apply:

- ROCm 7.0+
- Triton (required commit: [aafec417bded34db6308f5b3d6023daefae43905](https://github.com/triton-lang/triton/tree/aafec417bded34db6308f5b3d6023daefae43905) or later)

## Installation Methods
### 1. Direct Installation from Git

For a quick installation directly from the repository:

```shell
pip install git+https://github.com/ROCm/iris.git
```

### 2. Using Docker Compose

The easiest way to get started if you don't have the dependencies installed is using Docker Compose:

```shell
# Clone the repository
git clone https://github.com/ROCm/iris.git
cd iris

# Start the development container
docker compose up --build -d

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode
cd iris && pip install -e .
```

### 3. Manual Docker Setup

If you prefer to build and run Docker containers manually:

```shell
# Build the Docker image
./docker/build.sh

# Run the container
./docker/run.sh

# Install Iris in development mode
pip install -e .
```


### 4. Apptainer/Singularity

For HPC environments or systems where Docker is not available:

```shell
# Build the Apptainer image
./apptainer/build.sh

# Run the container
./apptainer/run.sh

# Install Iris in development mode
pip install -e .
```


## Next Steps

Once you have Iris running with any of these methods:

- Explore the [Examples](../reference/examples.md) directory
- Learn about the [Programming Model](../conceptual/programming-model.md)