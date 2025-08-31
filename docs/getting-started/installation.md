# Installation Guide

This guide covers how to install Iris on your system using various methods.

## Overview

Iris has minimal dependencies including Python, PyTorch, ROCm HIP runtime, MPI, and Triton. This guide will walk you through the installation process using different approaches.

## Prerequisites

### System Requirements

- Linux operating system (Ubuntu 22.04+)
- AMD GPU with ROCm 6.3.1+ support (MI300X, MI350X, MI355X, or other ROCm-compatible GPUs)
- Python 3.10+
- ROCm 6.3.1+ HIP runtime

### Required Software

**Minimum working requirements based on the Docker setup:**

- Python 3.10+
- PyTorch 2.0+ (ROCm version)
- ROCm 6.3.1+ HIP runtime
- OpenMPI
- Git
- CMake, Ninja, build-essential
- Triton (specific commit: dd5823453bcc7973eabadb65f9d827c43281c434)

**Note**: These versions represent the minimum working configuration. Using different versions may cause compatibility issues.

## Installation Methods

### 1. Using Docker Compose (Recommended)

The easiest way to get started is using Docker Compose:

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

### 2. Manual Docker Setup

If you prefer to build and run Docker containers manually:

```shell
# Build the Docker image
./docker/build.sh <image-name>

# Run the container
./docker/run.sh <image-name>

# Install Iris in development mode
pip install -e .
```

#### Docker Build Options

The build script accepts custom image names and can be customized:

```bash
# Build with custom name
./build.sh my-iris-image

# Build with specific ROCm version
export ROCM_VERSION=5.7.3
./build.sh iris-rocm-5.7.3
```

#### Docker Run Options

```bash
# Run with custom port mapping
./run.sh iris-dev -p 8888:8888

# Run with volume mounts
./run.sh iris-dev -v /path/to/data:/data

# Run in detached mode
./run.sh iris-dev -d
```

### 3. Apptainer/Singularity

For HPC environments or systems where Docker is not available:

```shell
# Build the Apptainer image
./apptainer/build.sh

# Run the container
./apptainer/run.sh

# Install Iris in development mode
pip install -e .
```

#### Apptainer Build Options

```bash
# Build with custom output name
./build.sh -o iris.sif

# Build with specific ROCm version
export ROCM_VERSION=5.7.3
./build.sh
```

#### Apptainer Run Options

```bash
# Run with custom working directory
./run.sh -W /workspace

# Run with GPU access
./run.sh --nv

# Run with custom environment
./run.sh --env-file .env
```

### 4. From Source (Manual Setup)

For advanced users who want full control over their environment:

**Note**: Manual setup is complex and requires exact dependency versions to match the working Docker setup.

#### Prerequisites Installation

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build

# CentOS/RHEL
sudo yum install -y \
    python3-devel \
    python3-pip \
    openmpi-devel \
    gcc \
    gcc-c++ \
    cmake \
    git \
    wget
```

#### Python Environment Setup

```bash
# Create virtual environment
python3 -m venv iris-env
source iris-env/bin/activate

# Install Python dependencies
pip install --upgrade pip wheel
pip install numpy requests mpi4py
```

#### ROCm and HIP Installation

Follow the [official ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) for your system.

#### Triton Installation

```bash
# Install Triton from specific commit
git clone https://github.com/triton-lang/triton.git /opt/triton
cd /opt/triton
git checkout dd5823453bcc7973eabadb65f9d827c43281c434
pip install -e .
export PYTHONPATH=/opt/triton
```

#### Iris Installation

```bash
# Clone and install Iris
git clone https://github.com/ROCm/iris.git
cd iris
pip install -e .
```

**Important**: The versions above represent the minimum working configuration. Using different versions may cause build failures.

## Verification

After installation, verify that Iris is working:

```python
import iris

# Test basic functionality
iris_ctx = iris.iris(heap_size=2**30)
print(f"Rank: {iris_ctx.get_rank()}, Size: {iris_ctx.get_num_ranks()}")
```

You can also run these verification commands:

```bash
# Check Iris import
python -c "import iris; print('Iris imported successfully')"

# Check MPI
mpirun -np 2 python -c "import iris; print('MPI + Iris working')"

# Check GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dependencies

**Minimum working dependencies based on the Docker setup:**

### Python Dependencies (Automatically Installed)
- `torch` (ROCm version)
- `triton` (from specific git commit dd5823453bcc7973eabadb65f9d827c43281c434)
- `mpi4py` (OpenMPI backend)
- `numpy`, `requests` (via PyTorch)

### System Dependencies (Must Be Installed)
- ROCm 6.3.1+ HIP runtime
- OpenMPI
- CMake, build-essential

**Note**: These represent the minimum working configuration. Using different versions may cause compatibility issues.

## Environment Variables

The containers set these important environment variables:

```bash
export TRITON_PATH=/opt/triton  # or /workspace/triton for Apptainer
export PYTHONPATH=$TRITON_PATH
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/openmpi/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm/bin:/usr/lib/openmpi/bin:$PATH
export OMPI_MCA_mtl="^ofi"
export OMPI_MCA_pml="ob1"
```

For optimal performance, you may also want to set:

```bash
# ROCm environment
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HSA_ENABLE_SDMA=0

# MPI environment
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Iris environment
export IRIS_HEAP_SIZE=1073741824  # 1GB in bytes
export IRIS_LOG_LEVEL=INFO
```

## Community & Support

### GitHub Discussions: Ask questions and share ideas
Join the [GitHub Discussions](https://github.com/ROCm/iris/discussions) to ask questions, share ideas, and connect with the Iris community.

### GitHub Issues: Report bugs and request features
Found a bug or have a feature request? Report it on [GitHub Issues](https://github.com/ROCm/iris/issues).

### Contributing: Help make Iris better for everyone
Want to contribute to Iris? Check out the [Contributing Guide](../reference/contributing.md) to learn how you can help make Iris better for everyone.

## Next Steps

Once you have Iris running with any of these methods:

- Follow the [Quick Start Guide](quick-start.md) to run your first example
- Explore the [Examples](../reference/examples.md) directory
- Learn about the [Programming Model](../conceptual/programming-model.md)

---