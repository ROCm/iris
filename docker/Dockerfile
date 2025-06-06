# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

FROM rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch

# Use bash shell for RUN commands
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV TRITON_PATH=/opt/triton \
    ROCM_PATH=/opt/rocm \
    OMPI_MCA_mtl="^ofi" \
    OMPI_MCA_pml="ob1"

ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:/usr/lib/openmpi/lib:$LD_LIBRARY_PATH \
    PATH="$ROCM_PATH/bin:/usr/lib/openmpi/bin:$PATH"

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget ninja-build cmake python3-pip python3-dev build-essential \
    openmpi-bin libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages with pip
RUN pip3 install --upgrade pip && \
    pip3 install wheel jupyter 
    
# This needs sudo, I can only get it to install with sudo
# or using conda, but conda runs into issues with too many requests.
#  https://stackoverflow.com/a/54052470/5729690
RUN sudo pip3 install mpi4py

# Clone and install Triton
WORKDIR $TRITON_PATH
RUN git clone https://github.com/triton-lang/triton.git $TRITON_PATH
RUN pip3 install -e .
ENV PYTHONPATH=$TRITON_PATH:$PYTHONPATH

# Install rocprofiler-systems
WORKDIR /workspace
RUN wget https://github.com/ROCm/rocprofiler-systems/releases/latest/download/rocprofiler-systems-install.py && \
    python3 ./rocprofiler-systems-install.py --prefix /opt/rocprofiler-systems --rocm 6.3 && \
    rm -f rocprofiler-systems-install.py

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'echo "Welcome to the ROCm-aware Docker image!"' >> /entrypoint.sh && \
    echo 'if [ $# -eq 0 ]; then' >> /entrypoint.sh && \
    echo '    exec /bin/bash' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    exec "$@"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /entrypoint.sh && exec bash"]