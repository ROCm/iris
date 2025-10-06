#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example demonstrating how to use Iris with CUDA backend for NVIDIA GPUs.

Backend can be configured in two ways:
1. Build-time (recommended):
   pip install git+https://github.com/ROCm/iris.git --config-settings backend=nvidia

2. Runtime (if not set at build time):
   export IRIS_BACKEND=cuda
   python cuda_backend_example.py
"""

import os
import sys

# Set CUDA backend if not configured at build time
# This must be done before importing iris to take effect
if "IRIS_BACKEND" not in os.environ:
    os.environ["IRIS_BACKEND"] = "cuda"

# Now import iris - it will use the CUDA backend
import iris


def main():
    """
    Demonstrate CUDA backend usage with Iris.

    This example shows:
    1. How to set the CUDA backend
    2. How to verify the backend is loaded
    3. Basic Iris operations with CUDA
    """

    print("=" * 60)
    print("Iris CUDA Backend Example")
    print("=" * 60)

    # Check which backend is being used
    try:
        backend = iris.hip.get_backend()
        print(f"✓ Backend loaded: {backend}")

        if backend == "cuda":
            print("✓ Successfully using CUDA backend for NVIDIA GPUs")
        else:
            print(f"! Note: Using {backend} backend instead of CUDA")
    except Exception as e:
        print(f"✗ Could not determine backend: {e}")

    # Initialize Iris with a symmetric heap
    heap_size = 1 << 30  # 1 GB
    print(f"\nInitializing Iris with {heap_size / (1024**3):.1f} GB heap...")

    try:
        ctx = iris.iris(heap_size)
        print("✓ Iris initialized successfully")
        print(f"  - Rank: {ctx.get_rank()}")
        print(f"  - Number of ranks: {ctx.get_num_ranks()}")
        print(f"  - Device: {ctx.get_device()}")
        print(f"  - Compute units: {ctx.get_cu_count()}")

        # Allocate a tensor on the symmetric heap
        print("\nAllocating tensor on symmetric heap...")
        tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
        print(f"✓ Tensor allocated: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  - On symmetric heap: {ctx._Iris__on_symmetric_heap(tensor)}")
        print(f"  - Device: {tensor.device}")

    except Exception as e:
        print(f"✗ Error initializing Iris: {e}")
        print("\nNote: This example requires:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - PyTorch with CUDA")
        print("  - NCCL for distributed operations")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import torch
    import torch.distributed as dist

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This example requires NVIDIA GPU.")
        print("\nTo use Iris with AMD GPUs, use the default HIP backend:")
        print("  python your_script.py  # No IRIS_BACKEND needed")
        sys.exit(1)

    # For this simple example, we'll run single-rank
    # For multi-rank examples, see the examples/ directory
    main()
