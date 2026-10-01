# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Shared handle serialization utilities for fabric drivers.

These helpers convert between raw fabric handles and uint8 tensors that higher
layers can exchange with torch.distributed collectives. Handle size validation,
when desired, is backend-specific and can be passed in explicitly.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

__all__ = [
    "fabric_handle_bytes_to_tensor",
    "fabric_tensor_to_handle_bytes",
]


def fabric_handle_bytes_to_tensor(
    handle_bytes: bytes,
    device: Union[torch.device, str],
    expected_num_bytes: Optional[int] = None,
) -> torch.Tensor:
    """Serialize a raw fabric handle into a uint8 tensor."""

    if expected_num_bytes is not None and len(handle_bytes) != expected_num_bytes:
        raise ValueError(f"Fabric handle must be {expected_num_bytes} bytes, got {len(handle_bytes)}")
    return torch.tensor(list(handle_bytes), dtype=torch.uint8, device=device)


def fabric_tensor_to_handle_bytes(handle_tensor: torch.Tensor, expected_num_bytes: Optional[int] = None) -> bytes:
    """Deserialize a uint8 tensor back into raw handle bytes."""

    flattened = handle_tensor.detach().flatten()
    if flattened.dtype != torch.uint8:
        raise ValueError("Fabric handle tensor must have dtype torch.uint8")
    if expected_num_bytes is not None and flattened.numel() != expected_num_bytes:
        raise ValueError(f"Fabric handle tensor must have {expected_num_bytes} elements, got {flattened.numel()}")
    return bytes(flattened.to("cpu", copy=True).tolist())
