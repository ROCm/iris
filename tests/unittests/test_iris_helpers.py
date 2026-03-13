# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import iris
from iris import tensor_creation


def test_get_device():
    shmem = iris.iris(1 << 20)

    # Test that get_device returns the correct device
    device = shmem.get_device()
    assert str(device) == shmem.device

    # Test that the device format is correct (should be "cuda:X")
    assert device.type == "cuda"
    assert device.index is not None


def test_device_validation():
    shmem = iris.iris(1 << 20)
    iris_device = shmem.get_device()

    # Test valid devices
    assert tensor_creation.is_valid_device(None, iris_device)  # None is always valid
    assert tensor_creation.is_valid_device(shmem.device, iris_device)  # Same device
    assert tensor_creation.is_valid_device(torch.device(shmem.device), iris_device)  # PyTorch device object

    # Test that "cuda" works for any CUDA device
    if shmem.device.startswith("cuda:"):
        assert tensor_creation.is_valid_device("cuda", iris_device)  # "cuda" should work for any CUDA device

    # Test invalid devices
    assert not tensor_creation.is_valid_device("cpu", iris_device)  # CPU is always invalid
    assert not tensor_creation.is_valid_device("mps", iris_device)  # MPS is always invalid

    # Test that different CUDA device indices are rejected
    if shmem.device.startswith("cuda:") and torch.cuda.device_count() >= 2:
        current_device = torch.device(shmem.device)
        different_cuda = f"cuda:{(current_device.index + 1) % torch.cuda.device_count()}"  # Use next GPU
        assert not tensor_creation.is_valid_device(different_cuda, iris_device)
