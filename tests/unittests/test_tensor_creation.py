# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the shared tensor_creation abstraction module.

These tests verify that the module-level helpers behave correctly in isolation
and that both Iris and IrisGluon delegate to them.
"""

import pytest
import torch

import iris
from iris import tensor_creation


# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------


def test_parse_size_flat_args():
    size, n = tensor_creation.parse_size((2, 3))
    assert size == (2, 3)
    assert n == 6


def test_parse_size_nested_tuple():
    size, n = tensor_creation.parse_size(((2, 3),))
    assert size == (2, 3)
    assert n == 6


def test_parse_size_nested_list():
    size, n = tensor_creation.parse_size(([4, 5],))
    assert size == [4, 5]
    assert n == 20


def test_parse_size_scalar():
    # zeros(()) → *size=((),) → flattened to ()
    size, n = tensor_creation.parse_size(((),))
    assert size == ()
    assert n == 1  # math.prod(()) == 1


def test_parse_size_single_int():
    size, n = tensor_creation.parse_size((7,))
    assert size == (7,)
    assert n == 7


# ---------------------------------------------------------------------------
# is_valid_device / throw_if_invalid_device
# ---------------------------------------------------------------------------


def test_is_valid_device_none():
    iris_device = torch.device("cuda:0")
    assert tensor_creation.is_valid_device(None, iris_device) is True


def test_is_valid_device_matching_cuda():
    iris_device = torch.device("cuda:0")
    assert tensor_creation.is_valid_device(torch.device("cuda:0"), iris_device) is True


def test_is_valid_device_cuda_no_index():
    iris_device = torch.device("cuda:0")
    assert tensor_creation.is_valid_device("cuda", iris_device) is True


def test_is_valid_device_cpu_invalid():
    iris_device = torch.device("cuda:0")
    assert tensor_creation.is_valid_device("cpu", iris_device) is False


def test_throw_if_invalid_device_raises():
    iris_device = torch.device("cuda:0")
    with pytest.raises(RuntimeError):
        tensor_creation.throw_if_invalid_device("cpu", iris_device)


def test_throw_if_invalid_device_ok():
    iris_device = torch.device("cuda:0")
    # Should not raise
    tensor_creation.throw_if_invalid_device(None, iris_device)
    tensor_creation.throw_if_invalid_device("cuda", iris_device)


# ---------------------------------------------------------------------------
# apply_layout
# ---------------------------------------------------------------------------


def test_apply_layout_strided_passthrough():
    t = torch.zeros(2, 3, device="cuda")
    result = tensor_creation.apply_layout(t, torch.strided)
    assert result is t


def test_apply_layout_unsupported_raises():
    t = torch.zeros(2, 3, device="cuda")
    with pytest.raises(ValueError):
        tensor_creation.apply_layout(t, torch.sparse_coo)


# ---------------------------------------------------------------------------
# throw_if_invalid_output_tensor  (uses a real heap via iris.iris)
# ---------------------------------------------------------------------------


def test_throw_if_invalid_output_tensor_wrong_numel():
    shmem = iris.iris(1 << 20)
    out = shmem._Iris__allocate(4, torch.float32)
    with pytest.raises(RuntimeError, match="4 elements, but 9 are required"):
        tensor_creation.throw_if_invalid_output_tensor(shmem.heap, out, 9, torch.float32)


def test_throw_if_invalid_output_tensor_wrong_dtype():
    shmem = iris.iris(1 << 20)
    out = shmem._Iris__allocate(9, torch.int32)
    with pytest.raises(RuntimeError, match="dtype torch.int32, but torch.float32 is required"):
        tensor_creation.throw_if_invalid_output_tensor(shmem.heap, out, 9, torch.float32)


def test_throw_if_invalid_output_tensor_not_symmetric():
    shmem = iris.iris(1 << 20)
    regular = torch.zeros(9, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="not on the symmetric heap"):
        tensor_creation.throw_if_invalid_output_tensor(shmem.heap, regular, 9, torch.float32)


def test_throw_if_invalid_output_tensor_ok():
    shmem = iris.iris(1 << 20)
    out = shmem._Iris__allocate(9, torch.float32)
    # Should not raise
    tensor_creation.throw_if_invalid_output_tensor(shmem.heap, out, 9, torch.float32)


# ---------------------------------------------------------------------------
# Creation functions via iris.iris (Triton backend) – delegation check
# ---------------------------------------------------------------------------


def test_zeros_via_iris():
    shmem = iris.iris(1 << 20)
    t = shmem.zeros(3, 4, dtype=torch.float32)
    assert t.shape == (3, 4)
    assert t.dtype == torch.float32
    assert torch.all(t == 0)
    assert shmem.is_symmetric(t)


def test_ones_via_iris():
    shmem = iris.iris(1 << 20)
    t = shmem.ones(2, 5, dtype=torch.float32)
    assert t.shape == (2, 5)
    assert torch.all(t == 1)
    assert shmem.is_symmetric(t)


def test_full_via_iris():
    shmem = iris.iris(1 << 20)
    t = shmem.full((3, 3), 7.0, dtype=torch.float32)
    assert t.shape == (3, 3)
    assert torch.all(t == 7.0)
    assert shmem.is_symmetric(t)


def test_zeros_like_via_iris():
    shmem = iris.iris(1 << 20)
    inp = shmem.ones(4, 2, dtype=torch.float16)
    t = shmem.zeros_like(inp)
    assert t.shape == inp.shape
    assert t.dtype == inp.dtype
    assert torch.all(t == 0)
    assert shmem.is_symmetric(t)


# ---------------------------------------------------------------------------
# Module-level zeros / ones / full / zeros_like  (direct calls)
# ---------------------------------------------------------------------------


def test_module_zeros():
    shmem = iris.iris(1 << 20)
    t = tensor_creation.zeros(shmem.heap, shmem.get_device(), (2, 3), dtype=torch.float32)
    assert t.shape == (2, 3)
    assert torch.all(t == 0)
    assert shmem.is_symmetric(t)


def test_module_ones():
    shmem = iris.iris(1 << 20)
    t = tensor_creation.ones(shmem.heap, shmem.get_device(), (4,), dtype=torch.int32)
    assert t.shape == (4,)
    assert torch.all(t == 1)
    assert shmem.is_symmetric(t)


def test_module_full():
    shmem = iris.iris(1 << 20)
    t = tensor_creation.full(shmem.heap, shmem.get_device(), (5,), 3.14, dtype=torch.float32)
    assert t.shape == (5,)
    assert shmem.is_symmetric(t)


def test_module_zeros_like():
    shmem = iris.iris(1 << 20)
    inp = shmem.ones(3, 3, dtype=torch.float64)
    t = tensor_creation.zeros_like(shmem.heap, shmem.get_device(), inp)
    assert t.shape == (3, 3)
    assert t.dtype == torch.float64
    assert torch.all(t == 0)
    assert shmem.is_symmetric(t)


# ---------------------------------------------------------------------------
# parse_size edge: empty tuple (scalar)
# ---------------------------------------------------------------------------


def test_zeros_scalar():
    shmem = iris.iris(1 << 20)
    t = shmem.zeros(())
    assert t.shape == ()
    assert t.numel() == 1
    assert t.item() == 0
    assert shmem.is_symmetric(t)
