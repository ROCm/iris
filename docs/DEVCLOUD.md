# AMD Developer Cloud Setup Guide

This guide provides step-by-step instructions for setting up Iris on the AMD Developer Cloud environment.

## Prerequisites

Before starting, ensure you have access to an AMD Developer Cloud and create a GPU Droplet.

## Environment Setup

### 1. Set ROCm Environment Variables

First, set up the ROCm environment variables:

```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib
```

**Note**: You may want to add these to your shell profile (`.bashrc`, `.zshrc`, etc.) for persistence across sessions.

### 2. Install System Dependencies

Install the required system packages:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv cmake openmpi-bin libopenmpi-dev
```

### 3. Create and Activate Virtual Environment

Create a Python virtual environment to isolate Iris dependencies:

```bash
# Create virtual environment
python3 -m venv iris_env

# Activate virtual environment
source iris_env/bin/activate
```

### 4. Install Python Dependencies
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```


## Iris Installation

### 1. Clone the Repository

```bash
git clone git@github.com:ROCm/iris.git
cd iris
```

### 2. Install Iris

Install Iris in development mode:

```bash
pip install -e .
```

Next, you can run the examples! See the [Examples README](../examples/README.md) for detailed information about available examples and how to run them.
