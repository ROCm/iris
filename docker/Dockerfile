# SPDX-License-Identifier: MIT
FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0

# Use bash shell for RUN commands
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV ROCM_PATH=/opt/rocm

ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH \
    PATH="$ROCM_PATH/bin:$PATH"

# Install system packages needed for Iris RDMA
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget cmake build-essential \
    libibverbs-dev librdmacm-dev \
    python3-pip python3-dev \
    infiniband-diags \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install pybind11

# Set working directory
WORKDIR /workspace

# Create entrypoint script
RUN printf '#!/bin/bash\n\
echo "=== Iris RDMA Development Environment ==="\n\
echo "ROCm version: $(cat $ROCM_PATH/.info/version 2>/dev/null || echo unknown)"\n\
echo "PyTorch version: $(python -c '\''import torch; print(torch.__version__)'\'' 2>/dev/null)"\n\
\n\
# GPU detection using PyTorch\n\
python -c '\''\n\
import torch\n\
if torch.cuda.is_available():\n\
    count = torch.cuda.device_count()\n\
    print(f"GPUs available: {count}")\n\
    for i in range(count):\n\
        name = torch.cuda.get_device_name(i)\n\
        print(f"  GPU[{i}]: {name}")\n\
else:\n\
    print("GPUs available: 0")\n\
'\'' 2>/dev/null || echo "GPUs available: 0"\n\
\n\
# InfiniBand detection\n\
if [ -d /dev/infiniband ]; then\n\
    IB_COUNT=$(ls /dev/infiniband/uverbs* 2>/dev/null | wc -l)\n\
    echo "InfiniBand devices available: $IB_COUNT"\n\
    if [ $IB_COUNT -gt 0 ]; then\n\
        echo "InfiniBand device(s): $(ls /sys/class/infiniband/ 2>/dev/null | tr '\''\n'\'' '\'' '\'')"\n\
    fi\n\
else\n\
    echo "InfiniBand devices available: 0"\n\
fi\n\
echo "======================================"\n\
if [ $# -eq 0 ]; then\n\
    exec /bin/bash\n\
else\n\
    exec "$@"\n\
fi\n' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

