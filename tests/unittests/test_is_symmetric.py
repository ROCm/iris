# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test is_symmetric() public API.
"""

import torch
import iris


def test_is_symmetric_basic():
    """Test basic is_symmetric() functionality."""
    ctx = iris.iris(1 << 20, allocator_type="torch")

    # Tensor allocated by ctx should be on symmetric heap
    symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
    assert ctx.is_symmetric(symmetric_tensor)

    # External tensor should not be on symmetric heap
    external_tensor = torch.zeros(1000, dtype=torch.float32, device="cuda")
    assert not ctx.is_symmetric(external_tensor)


def test_is_symmetric_imported_tensor():
    """Test is_symmetric() with imported external tensor (vmem allocator)."""
    ctx = iris.iris(64 << 20, allocator_type="vmem")

    # External tensor should not be on symmetric heap
    external_tensor = torch.randn(500, dtype=torch.float32, device="cuda")
    assert not ctx.is_symmetric(external_tensor)

    # After import, tensor should be on symmetric heap
    imported_tensor = ctx.as_symmetric(external_tensor)
    assert ctx.is_symmetric(imported_tensor)

    # Original external tensor still not on symmetric heap
    assert not ctx.is_symmetric(external_tensor)
