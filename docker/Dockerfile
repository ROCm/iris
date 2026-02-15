# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.13_pytorch_release_2.9.1

# Use bash shell for RUN commands
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV TRITON_PATH=/opt/triton \
    ROCM_PATH=/opt/rocm \
    OMPI_MCA_mtl="^ofi" \
    OMPI_MCA_pml="ob1" \
    HSA_NO_SCRATCH_RECLAIM=1

ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH \
    PATH="$ROCM_PATH/bin:$PATH"

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget ninja-build cmake python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r video 2>/dev/null || true && \
    groupadd -r render 2>/dev/null || true

# Install Python packages with pip
RUN pip3 install --upgrade pip && \
    pip3 install wheel jupyter

# Clone and install Triton
WORKDIR $TRITON_PATH
RUN git clone https://github.com/triton-lang/triton.git $TRITON_PATH
RUN git checkout bcbcabdd0cff6539c7168299075992b2a23ff38e
RUN pip3 install -e .
ENV PYTHONPATH=$TRITON_PATH

# Set up workspace
WORKDIR /workspace

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