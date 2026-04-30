# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Ragged tensor metadata for grouped expert computation.

Simplified port of triton_kernels/tensor_details/ragged_tensor.py:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/ragged_tensor.py

Only the fields needed by the simplified grouped matmul are retained:
slice_sizes, slice_offs, and n_slices.
"""

import torch
from dataclasses import dataclass


@dataclass
class RaggedTensorMetadata:
    """Lightweight ragged tensor descriptor.

    Example with 4 experts receiving [3, 0, 5, 2] tokens:
        slice_sizes = [3, 0, 5, 2]
        slice_offs  = [0, 3, 3, 8, 10]
    """

    slice_sizes: torch.Tensor  # (n_slices,) int32
    slice_offs: torch.Tensor  # (n_slices + 1,) int32

    @property
    def n_slices(self) -> int:
        return self.slice_sizes.shape[0]


def make_ragged_tensor_metadata(
    slice_sizes: torch.Tensor,
    n_total_rows: int,
) -> RaggedTensorMetadata:
    """Build ragged metadata from per-expert token counts.

    Args:
        slice_sizes: (n_experts,) int32 tensor of token counts per expert.
        n_total_rows: total number of active token-expert slots (for validation).
    """
    assert slice_sizes.ndim == 1
    slice_sizes = slice_sizes.to(torch.int32)
    offs = torch.zeros(slice_sizes.shape[0] + 1, dtype=torch.int32, device=slice_sizes.device)
    offs[1:] = torch.cumsum(slice_sizes, dim=0)
    return RaggedTensorMetadata(slice_sizes, offs)


def remap_ragged_tensor_metadata(
    metadata: RaggedTensorMetadata,
    expt_map: torch.Tensor,
) -> RaggedTensorMetadata:
    """Remap global expert metadata to a local expert view.

    expt_map: (n_expts_tot,) int32 where expt_map[global_id] is the local id
              on this rank, or -1 if the expert is not on this rank.

    Returns metadata containing only the experts owned by this rank, with
    ORIGINAL global offsets preserved so the grouped matmul addresses the
    correct positions in the globally-indexed dispatch buffer.
    """
    valid = expt_map != -1
    local_ids = expt_map[valid]
    n_local = int(local_ids.max().item()) + 1 if local_ids.numel() > 0 else 0
    device = metadata.slice_sizes.device
    local_sizes = torch.zeros(n_local, dtype=torch.int32, device=device)
    local_offs = torch.zeros(n_local + 1, dtype=torch.int32, device=device)
    for g in range(expt_map.shape[0]):
        lid = expt_map[g].item()
        if lid >= 0:
            local_sizes[lid] = metadata.slice_sizes[g]
            local_offs[lid] = metadata.slice_offs[g]
    if n_local > 0:
        local_offs[n_local] = local_offs[n_local - 1] + local_sizes[n_local - 1]
    return RaggedTensorMetadata(local_sizes, local_offs)
