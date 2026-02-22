# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tensor creation abstraction for symmetric-heap tensors.

Provides shared helpers (parse_size, device validation, allocation wiring,
output-tensor validation, layout and memory-format handling) and the core
creation logic for ``zeros``, ``ones``, ``full``, and ``zeros_like``.

Both the Triton :class:`~iris.iris.Iris` backend and the Gluon
:class:`~iris.experimental.iris_gluon.IrisGluon` backend delegate to these
functions so that the logic lives in exactly one place.
"""

import math

import torch


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def parse_size(size):
    """Parse a *size* argument and return ``(size_tuple, num_elements)``.

    Handles the common calling conventions::

        zeros(2, 3)         # *size = (2, 3)
        zeros((2, 3))       # *size = ((2, 3),)
        zeros([2, 3])       # *size = ([2, 3],)
        zeros(((2, 3),))    # nested – flattened once
    """
    # Flatten one level of wrapping tuple/list
    while len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    num_elements = math.prod(size)
    return size, num_elements


def is_valid_device(device, iris_device) -> bool:
    """Return *True* when *device* is compatible with *iris_device*.

    Args:
        device: Requested device (``str``, :class:`torch.device`, or ``None``).
            ``None`` is treated as "use the Iris default" and is always valid.
        iris_device (:class:`torch.device`): Device of the Iris symmetric heap.
    """
    if device is None:
        return True  # None means use default device

    requested_device = torch.device(device) if isinstance(device, str) else device

    # Both must be CUDA devices; index must match (or requested has no index)
    if requested_device.type == "cuda" and iris_device.type == "cuda":
        if requested_device.index is None:
            return True
        return requested_device.index == iris_device.index

    # Non-CUDA devices are not supported
    return False


def throw_if_invalid_device(device, iris_device):
    """Raise :exc:`RuntimeError` when *device* is incompatible with *iris_device*.

    Args:
        device: Requested device (``str``, :class:`torch.device`, or ``None``).
        iris_device (:class:`torch.device`): Device of the Iris symmetric heap.

    Raises:
        RuntimeError: If the device does not match the Iris instance device.
    """
    if not is_valid_device(device, iris_device):
        raise RuntimeError(
            f"Device mismatch: requested device {device} but Iris instance is on device {iris_device}. "
            f"Iris only supports tensors on its own device."
        )


def throw_if_invalid_output_tensor(heap, tensor: torch.Tensor, num_elements: int, dtype: torch.dtype):
    """Validate that *tensor* is suitable as an output buffer.

    Checks element count, dtype, and symmetric-heap membership in that order.

    Args:
        heap: Symmetric heap instance exposing ``is_symmetric(tensor)``.
        tensor (:class:`torch.Tensor`): Candidate output tensor.
        num_elements (int): Required number of elements.
        dtype (:class:`torch.dtype`): Required dtype.

    Raises:
        RuntimeError: On any mismatch.
    """
    if tensor.numel() != num_elements:
        raise RuntimeError(f"The output tensor has {tensor.numel()} elements, but {num_elements} are required")
    if tensor.dtype != dtype:
        raise RuntimeError(f"The output tensor has dtype {tensor.dtype}, but {dtype} is required")
    if not heap.is_symmetric(tensor):
        raise RuntimeError("The output tensor is not on the symmetric heap")


def apply_layout(tensor: torch.Tensor, layout: torch.layout) -> torch.Tensor:
    """Return *tensor* after applying *layout*.

    Only :data:`torch.strided` is currently supported.

    Raises:
        ValueError: For unsupported layouts.
    """
    if layout == torch.strided:
        return tensor
    raise ValueError(f"Layout {layout} not supported. Only torch.strided is currently supported.")


# ---------------------------------------------------------------------------
# Memory-format helper (used by zeros_like)
# ---------------------------------------------------------------------------


def _create_tensor_with_strides(allocate_fn, original_tensor: torch.Tensor, size: tuple, strides: tuple):
    """Allocate a symmetric-heap tensor with the given *size* and *strides*.

    Creates a temporary tensor to establish the desired layout, copies data
    from *original_tensor* (with any necessary permutation), then returns a
    view of a freshly heap-allocated buffer with the requested strides.

    Args:
        allocate_fn: Callable ``(num_elements, dtype) -> torch.Tensor`` that
            allocates flat storage on the symmetric heap.
        original_tensor (:class:`torch.Tensor`): Source tensor (contiguous).
        size (tuple): Target shape.
        strides (tuple): Target strides.
    """
    temp_tensor = torch.empty_strided(size, strides, dtype=original_tensor.dtype, device=original_tensor.device)

    if size != original_tensor.shape:
        if len(size) == 4:
            N, H, W, C = size[0], size[1], size[2], size[3]
            expected_strides = (H * W * C, 1, W * C, C)
            if strides == expected_strides:
                permuted = original_tensor.permute(0, 2, 3, 1)
            else:
                try:
                    permuted = original_tensor.reshape(size)
                except Exception:
                    raise ValueError(
                        "Cannot safely permute or reshape tensor: size differs from original shape for unknown reason."
                    )
        elif len(size) == 5:
            N, D, H, W, C = size[0], size[1], size[2], size[3], size[4]
            expected_strides = (D * H * W * C, 1, H * W * C, W * C, C)
            if strides == expected_strides:
                permuted = original_tensor.permute(0, 2, 3, 4, 1)
            else:
                try:
                    permuted = original_tensor.reshape(size)
                except Exception:
                    raise ValueError(
                        "Cannot safely permute or reshape tensor: size differs from original shape for unknown reason."
                    )
        else:
            try:
                permuted = original_tensor.reshape(size)
            except Exception:
                raise ValueError(
                    "Cannot safely permute or reshape tensor: size differs from original shape for unknown reason."
                )
    else:
        permuted = original_tensor

    temp_tensor.copy_(permuted)

    num_elements = math.prod(size)
    heap_tensor = allocate_fn(num_elements, original_tensor.dtype)
    heap_tensor = heap_tensor.reshape(size)
    heap_tensor.copy_(temp_tensor)
    del temp_tensor

    return torch.as_strided(heap_tensor, size, strides)


def apply_memory_format(
    allocate_fn,
    tensor: torch.Tensor,
    size: tuple,
    memory_format: torch.memory_format,
    input_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """Apply *memory_format* to *tensor*, keeping it on the symmetric heap.

    Args:
        allocate_fn: Callable ``(num_elements, dtype) -> torch.Tensor`` for
            heap allocation (used when a new stride layout requires a copy).
        tensor (:class:`torch.Tensor`): Tensor to reformat.
        size (tuple): Shape of *tensor*.
        memory_format (:class:`torch.memory_format`): Desired memory format.
        input_tensor (:class:`torch.Tensor`, optional): Reference tensor used
            to detect the format to preserve when
            *memory_format* is :data:`torch.preserve_format`.

    Returns:
        :class:`torch.Tensor`: Tensor in the requested memory format.
    """
    if memory_format == torch.contiguous_format:
        return tensor

    if memory_format == torch.channels_last and len(size) == 4:
        N, C, H, W = size[0], size[1], size[2], size[3]
        return _create_tensor_with_strides(allocate_fn, tensor, size, (C * H * W, 1, C * W, C))

    if memory_format == torch.channels_last_3d and len(size) == 5:
        N, C, D, H, W = size[0], size[1], size[2], size[3], size[4]
        return _create_tensor_with_strides(allocate_fn, tensor, size, (C * D * H * W, 1, C * D * W, C * W, C))

    if memory_format == torch.preserve_format:
        if input_tensor is not None:
            input_strides = input_tensor.stride()
            if len(size) == 4 and len(input_strides) == 4 and input_strides[1] == 1:
                input_shape = input_tensor.shape
                if len(input_shape) == 4:
                    return _create_tensor_with_strides(allocate_fn, tensor, input_shape, input_strides)
            elif len(size) == 5 and len(input_strides) == 5 and input_strides[1] == 1:
                input_shape = input_tensor.shape
                if len(input_shape) == 5:
                    return _create_tensor_with_strides(allocate_fn, tensor, input_shape, input_strides)
        return tensor

    # Unsupported format or dimension combination – fall back to contiguous
    return tensor


# ---------------------------------------------------------------------------
# Tensor creation functions
# ---------------------------------------------------------------------------


def zeros(heap, iris_device, size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """Allocate a zero-filled tensor on *heap*.

    Args:
        heap: Symmetric heap (``allocate`` / ``is_symmetric``).
        iris_device (:class:`torch.device`): Device of the heap.
        size (tuple): Shape of the tensor.

    Keyword Args:
        out (:class:`torch.Tensor`, optional): Pre-allocated output tensor.
        dtype (:class:`torch.dtype`, optional): Defaults to
            :func:`torch.get_default_dtype`.
        layout (:class:`torch.layout`): Default :data:`torch.strided`.
        device: Must be compatible with *iris_device* or ``None``.
        requires_grad (bool): Default ``False``.

    Returns:
        :class:`torch.Tensor`: Zero tensor on the symmetric heap.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = iris_device
    throw_if_invalid_device(device, iris_device)
    size, num_elements = parse_size(size)

    if out is not None:
        throw_if_invalid_output_tensor(heap, out, num_elements, dtype)
        out.zero_()
        tensor = out.view(size)
    else:
        tensor = heap.allocate(num_elements, dtype)
        tensor.zero_()
        tensor = tensor.reshape(size)

    tensor = apply_layout(tensor, layout)
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def ones(heap, iris_device, size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """Allocate a ones-filled tensor on *heap*.

    Args:
        heap: Symmetric heap (``allocate`` / ``is_symmetric``).
        iris_device (:class:`torch.device`): Device of the heap.
        size (tuple): Shape of the tensor.

    Keyword Args:
        out (:class:`torch.Tensor`, optional): Pre-allocated output tensor.
        dtype (:class:`torch.dtype`, optional): Defaults to
            :func:`torch.get_default_dtype`.
        layout (:class:`torch.layout`): Default :data:`torch.strided`.
        device: Must be compatible with *iris_device* or ``None``.
        requires_grad (bool): Default ``False``.

    Returns:
        :class:`torch.Tensor`: Ones tensor on the symmetric heap.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = iris_device
    throw_if_invalid_device(device, iris_device)
    size, num_elements = parse_size(size)

    if out is not None:
        throw_if_invalid_output_tensor(heap, out, num_elements, dtype)
        out.fill_(1)
        tensor = out.view(size)
    else:
        tensor = heap.allocate(num_elements, dtype)
        tensor.fill_(1)
        tensor = tensor.reshape(size)

    tensor = apply_layout(tensor, layout)
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def full(
    heap,
    iris_device,
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
):
    """Allocate a tensor filled with *fill_value* on *heap*.

    Args:
        heap: Symmetric heap (``allocate`` / ``is_symmetric``).
        iris_device (:class:`torch.device`): Device of the heap.
        size (tuple): Shape of the tensor.
        fill_value (scalar): Value to fill with.

    Keyword Args:
        out (:class:`torch.Tensor`, optional): Pre-allocated output tensor.
        dtype (:class:`torch.dtype`, optional): Inferred from *fill_value*
            when ``None``.
        layout (:class:`torch.layout`): Default :data:`torch.strided`.
        device: Must be compatible with *iris_device* or ``None``.
        requires_grad (bool): Default ``False``.

    Returns:
        :class:`torch.Tensor`: Filled tensor on the symmetric heap.
    """
    if dtype is None:
        if isinstance(fill_value, float):
            dtype = torch.get_default_dtype()
        elif isinstance(fill_value, int):
            dtype = torch.int64
        else:
            dtype = torch.get_default_dtype()
    if device is None:
        device = iris_device
    throw_if_invalid_device(device, iris_device)
    size, num_elements = parse_size(size)

    if out is not None:
        throw_if_invalid_output_tensor(heap, out, num_elements, dtype)
        out.fill_(fill_value)
        tensor = out.view(size)
    else:
        tensor = heap.allocate(num_elements, dtype)
        tensor.fill_(fill_value)
        tensor = tensor.reshape(size)

    tensor = apply_layout(tensor, layout)
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def zeros_like(
    heap,
    iris_device,
    input: torch.Tensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format,
):
    """Allocate a zero-filled tensor with the same shape as *input* on *heap*.

    Args:
        heap: Symmetric heap (``allocate`` / ``is_symmetric``).
        iris_device (:class:`torch.device`): Device of the heap.
        input (:class:`torch.Tensor`): Reference tensor.

    Keyword Args:
        dtype (:class:`torch.dtype`, optional): Defaults to ``input.dtype``.
        layout (:class:`torch.layout`, optional): Defaults to ``input.layout``.
        device: Defaults to ``input.device``; must be compatible with
            *iris_device*.
        requires_grad (bool): Default ``False``.
        memory_format (:class:`torch.memory_format`): Default
            :data:`torch.preserve_format`.

    Returns:
        :class:`torch.Tensor`: Zero tensor on the symmetric heap.
    """
    if dtype is None:
        dtype = input.dtype
    if layout is None:
        layout = input.layout
    if device is None:
        device = input.device
    throw_if_invalid_device(device, iris_device)

    size = input.size()
    num_elements = input.numel()

    new_tensor = heap.allocate(num_elements, dtype)
    new_tensor.zero_()
    new_tensor = new_tensor.reshape(size)

    new_tensor = apply_memory_format(heap.allocate, new_tensor, size, memory_format, input)
    new_tensor = apply_layout(new_tensor, layout)

    if requires_grad:
        new_tensor.requires_grad_()
    return new_tensor
