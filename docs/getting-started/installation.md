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

For scheduler-managed clusters, see the [SLURM guide](slurm.md).


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


## Optional: SDMA copy engine

Shader load/store, atomics, and collectives work with the base install. The SDMA
copy engine (`use_copy_engine=True` in device code, and host-initiated `put`,
`put_tile`, `put_tiles`, and `quiet`) needs [rocm-xio](https://github.com/ROCm/rocm-xio),
which is an optional extra:

```shell
pip install "iris[sdma] @ git+https://github.com/ROCm/iris.git"
# or, from a checkout
pip install -e ".[sdma]"
```

To add SDMA to an existing Iris install, install rocm-xio on its own. Use the
commit pinned by the `sdma` extra in `pyproject.toml`, since Iris builds SDMA
packets against that rocm-xio version:

```shell
pip install --force-reinstall --no-deps "rocm-xio @ git+https://github.com/ROCm/rocm-xio.git@cbe97e6392066bef7901121965ffadad19404da4"
```

`--force-reinstall` makes pip rebuild rocm-xio even if another commit is already
installed, since every commit reports the same package version. `--no-deps` keeps
pip from reinstalling unrelated packages such as PyTorch. Iris itself does not need
to be reinstalled; the next `iris.iris()` picks rocm-xio up.

The `enable_copy_engine` argument of `iris.iris()` controls whether SDMA queues are
initialized:

| `enable_copy_engine` | Result |
|---|---|
| `None` (default) | SDMA on only if rocm-xio is installed |
| `True` | SDMA on; `ImportError` if rocm-xio is not installed |
| `False` | SDMA off; rocm-xio is never imported |

Code that only uses shader load/store should pass `enable_copy_engine=False`, so it
does not initialize SDMA queues in environments that happen to have rocm-xio. With
SDMA off, `get_copy_engine_ctx()` returns `None` and the host SDMA APIs raise
`RuntimeError`.

## Selecting the AMD runtime for fabric communication

When multiple ROCm versions are installed, set `LD_LIBRARY_PATH` before starting
Python so it points to the SDK used by PyTorch. The AMD fabric driver first loads
`libamdhip64.so` and `libamd_smi.so` through that search path, then falls back to
system-library discovery if those names are unavailable. This prevents a stale
system SONAME from selecting a different HIP major version alongside PyTorch's
runtime. Mixing HIP runtimes in one process can abort during device initialization.

## Next Steps

Once you have Iris running with any of these methods:

- Explore the [Examples](../reference/examples.md) directory
- Learn about the [Programming Model](../conceptual/programming-model.md)
- For batch-scheduled environments, see [Running Iris on SLURM](slurm.md)
