# Setup Alternatives

This guide describes alternative ways to set up and run Iris, including manual Docker setup and Apptainer. Use these methods if Docker Compose is not suitable for your workflow.

## Manual Docker Setup

If you prefer to build and run Docker containers manually instead of using Docker Compose:

```bash
# Build the Docker image
cd docker
./build.sh iris-dev

# Run the container
./run.sh iris-dev

# Install Iris in development mode
pip install -e .
```

### Docker Build Options

The build script accepts custom image names and can be customized:

```bash
# Build with custom name
./build.sh my-iris-image

# Build with specific ROCm version
export ROCM_VERSION=5.7.3
./build.sh iris-rocm-5.7.3
```

### Docker Run Options

```bash
# Run with custom port mapping
./run.sh iris-dev -p 8888:8888

# Run with volume mounts
./run.sh iris-dev -v /path/to/data:/data

# Run in detached mode
./run.sh iris-dev -d
```

## Apptainer/Singularity

For HPC environments or systems where Docker is not available, use Apptainer:

```bash
# Build the Apptainer image
cd apptainer
./build.sh

# Run the container
./run.sh

# Install Iris in development mode
pip install -e .
```

### Apptainer Build Options

```bash
# Build with custom output name
./build.sh -o iris.sif

# Build with specific ROCm version
export ROCM_VERSION=5.7.3
./build.sh
```

### Apptainer Run Options

```bash
# Run with custom working directory
./run.sh -W /workspace

# Run with GPU access
./run.sh --nv

# Run with custom environment
./run.sh --env-file .env
```

## Manual System Installation

For advanced users who want full control over their environment:

### Prerequisites Installation

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    cmake

# CentOS/RHEL
sudo yum install -y \
    python3-devel \
    python3-pip \
    openmpi-devel \
    gcc \
    gcc-c++ \
    cmake
```

### Python Environment Setup

```bash
# Create virtual environment
python3 -m venv iris-env
source iris-env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install numpy requests mpi4py
```

### ROCm and HIP Installation

Follow the [official ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) for your system.

### Triton Installation

```bash
# Install Triton from source (specific commit for compatibility)
pip install git+https://github.com/triton-lang/triton.git@dd5823453bcc7973eabadb65f9d827c43281c434
```

### Iris Installation

```bash
# Clone and install Iris
git clone https://github.com/ROCm/iris.git
cd iris
pip install -e .
```

## Environment Variables

Set these environment variables for optimal performance:

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

## Verification

After any setup method, verify your installation:

```bash
# Check Iris import
python -c "import iris; print('Iris imported successfully')"

# Check MPI
mpirun -np 2 python -c "import iris; print('MPI + Iris working')"

# Check GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

1. **Permission denied**: Use `sudo` or check file permissions
2. **ROCm not found**: Verify ROCm installation and PATH
3. **MPI errors**: Check OpenMPI/MPICH installation
4. **GPU not visible**: Check `rocm-smi` and device permissions

### Getting Help

- Check the [Installation Guide](installation.md) for basic setup
- Review the [Troubleshooting](../how-to/debug-common-issues.md) section
- Open an issue on GitHub with detailed error messages

## Next Steps

Once you have Iris running with any of these methods:

- Follow the [Quick Start Guide](quick-start.md) to run your first example
- Explore the [Examples](../reference/examples.md) directory
- Learn about the [Programming Model](../conceptual/programming-model.md)

---

**Need help with a specific setup method? Check the [Troubleshooting](../how-to/debug-common-issues.md) guide or start a discussion in GitHub Discussions!**
