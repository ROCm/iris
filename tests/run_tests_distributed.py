#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Worker script for running pytest tests under torchrun.
This script is invoked by torchrun and runs pytest within a distributed process group.
"""

import os
import sys

# Set required environment variable for RCCL on ROCm
os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")

import torch
import torch.distributed as dist

# torchrun sets these environment variables automatically
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Set the correct GPU for this specific process
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# Initialize distributed - torchrun already set up the environment
dist.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size,
    device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
)

try:
    # Import and run pytest with command-line arguments
    import pytest

    # Pass through all command-line arguments to pytest
    exit_code = pytest.main(sys.argv[1:])
    sys.exit(exit_code)
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
