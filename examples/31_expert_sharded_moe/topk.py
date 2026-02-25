# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Top-k expert routing for MoE.

Ported / simplified from triton_kernels/topk.py and bitmatrix.py:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/topk.py
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py

Provides:
  - PyTorch-based top-k + softmax (matches topk_torch reference)
  - Host-side BitmatrixMetadata (col_sum, row_sorted_indx, col_sorted_indx)
  - A convenience ``topk`` function
"""

import torch
from dataclasses import dataclass


@dataclass
class BitmatrixMetadata:
    """Routing indices derived from the top-k selection.

    col_sum:          (n_expts,)        histogram: tokens per expert
    row_sorted_indx:  (n_tokens * k,)   flat token-expert slots grouped by expert (dispatch order)
    col_sorted_indx:  (n_tokens * k,)   inverse permutation (combine order)
    """

    col_sum: torch.Tensor
    row_sorted_indx: torch.Tensor
    col_sorted_indx: torch.Tensor


@dataclass
class TopkResult:
    vals: torch.Tensor  # (n_tokens, k) softmax gating weights
    indx: torch.Tensor  # (n_tokens, k) expert indices (int16)
    mask_metadata: BitmatrixMetadata


# ---------------------------------------------------------------------------
# Host-side bitmatrix metadata construction (torch reference)
# ---------------------------------------------------------------------------


def _make_bitmatrix_metadata(indx: torch.Tensor, n_expts: int) -> BitmatrixMetadata:
    """Build dispatch/combine indices from the (n_tokens, k) expert-index tensor.

    Follows triton_kernels/tensor_details/bitmatrix.py (optimised convention):
      col_sorted_indx[expert_sorted_pos] = original flat index
      row_sorted_indx[original_flat_idx]  = expert_sorted_pos

    Handles -1 (invalid) entries correctly.
    """
    device = indx.device
    flat_indx = indx.reshape(-1).to(torch.int32)
    n_elements = flat_indx.numel()

    valid = flat_indx >= 0
    n_valid = valid.sum().item()

    col_sum = torch.histc(
        flat_indx[valid].float(),
        bins=n_expts,
        min=0,
        max=n_expts - 1,
    ).to(torch.int32)

    col_sorted_indx = torch.full((n_elements,), -1, dtype=torch.int32, device=device)
    row_sorted_indx = torch.full((n_elements,), -1, dtype=torch.int32, device=device)

    sort_keys = flat_indx.clone().long()
    sort_keys[~valid] = n_expts
    sorted_order = torch.argsort(sort_keys, stable=True).to(torch.int32)

    col_sorted_indx[:n_valid] = sorted_order[:n_valid]
    expert_positions = torch.arange(n_valid, device=device, dtype=torch.int32)
    row_sorted_indx.scatter_(0, sorted_order[:n_valid].long(), expert_positions)

    return BitmatrixMetadata(
        col_sum=col_sum,
        col_sorted_indx=col_sorted_indx,
        row_sorted_indx=row_sorted_indx,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def topk(
    x: torch.Tensor,
    k: int,
    apply_softmax: bool = True,
) -> TopkResult:
    """Compute top-k routing over expert logits.

    Uses PyTorch ops (matches upstream topk_torch reference).

    Args:
        x: (n_tokens, n_expts) float32 logit tensor.
        k: number of experts to activate per token.
        apply_softmax: whether to softmax the selected values.

    Returns:
        TopkResult with vals, indx, and mask_metadata.
    """
    n_tokens, n_expts = x.shape

    vals, indx = torch.topk(x.float(), k, dim=1, sorted=True)

    if apply_softmax:
        vals = torch.softmax(vals, dim=-1).to(x.dtype)
    else:
        vals = vals.to(x.dtype)
    indx = indx.to(torch.int16)

    mask_metadata = _make_bitmatrix_metadata(indx.to(torch.int32), n_expts)
    return TopkResult(vals=vals, indx=indx, mask_metadata=mask_metadata)
