# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0

# Install dependencies and Triton
RUN apt-get update && apt-get install -y git && \
    export TRITON_PATH=/workspace/triton && \
    git clone https://github.com/triton-lang/triton.git $TRITON_PATH && \
    cd $TRITON_PATH && \
    git checkout aafec417bded34db6308f5b3d6023daefae43905 && \
    pip install -e . && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV TRITON_PATH=/workspace/triton
ENV PYTHONPATH=${TRITON_PATH}/python/
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/conda/envs/py_3.10/bin:/opt/rocm/bin:${PATH}
ENV OMPI_MCA_mtl="^ofi"
ENV OMPI_MCA_pml="ob1"

# Set working directory
WORKDIR /workspace

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Welcome to the ROCm-aware Docker image!"\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

