# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
MoE routing utilities for expert-parallel dispatch/combine.

Promoted from examples/31_expert_sharded_moe/ into iris.ccl for production use.
Provides expert assignment, ragged tensor metadata, top-k routing, and reduce.

Ported from triton_kernels:
  - distributed.py (ExptAssignment)
  - tensor_details/ragged_tensor.py (RaggedTensorMetadata)
  - topk.py / bitmatrix.py (TopkResult, BitmatrixMetadata)
  - reduce.py (masked sum-reduce)
"""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Expert Assignment
# ---------------------------------------------------------------------------


@dataclass
class ExptAssignment:
    """Expert-to-rank assignment for expert-parallel MoE.

    Attributes:
        expt_bitmask: (n_shards, ceil(n_expts_tot / 32)) packed int32 bitmask.
            (expt_bitmask[i, j//32] >> j%32) & 1 == 1 iff expert j is owned by shard i.
        expt_boolmask: (n_shards, n_expts_tot) boolean mask.
        expt_map: (n_shards, n_expts_tot) local expert id or -1.
        n_expts_per_shard: list of expert counts per shard.
    """

    expt_bitmask: torch.Tensor
    expt_boolmask: torch.Tensor
    expt_map: torch.Tensor
    n_expts_per_shard: list[int]


def make_expt_dict_uniform(n_shards: int, n_expts_tot: int) -> dict[int, list[int]]:
    """Contiguous assignment: shard i owns experts [i*E_per_shard, (i+1)*E_per_shard)."""
    assert n_expts_tot % n_shards == 0, "n_expts_tot must be divisible by n_shards"
    e_per_shard = n_expts_tot // n_shards
    return {i: list(range(i * e_per_shard, (i + 1) * e_per_shard)) for i in range(n_shards)}


def make_expt_assignment(
    n_shards: int,
    n_expts_tot: int,
    expt_dict: dict[int, list[int]],
    device,
) -> ExptAssignment:
    """Build bitmask, boolmask, and local-id map from an expert ownership dict."""
    words = (n_expts_tot + 31) // 32
    expt_bitmask = torch.zeros((n_shards, words), dtype=torch.int32)
    expt_boolmask = torch.zeros((n_shards, n_expts_tot), dtype=torch.bool)
    counts = {e: 0 for e in range(n_expts_tot)}

    for shard, experts in expt_dict.items():
        if not (0 <= shard < n_shards):
            raise ValueError(f"shard {shard} out of range [0, {n_shards})")
        if len(experts) == 0:
            raise ValueError(f"shard {shard} has no experts")
        for e in experts:
            counts[e] += 1
            if not (0 <= e < n_expts_tot):
                raise ValueError(f"expert id {e} out of range [0, {n_expts_tot})")
            word = e >> 5
            bit = e & 31
            expt_bitmask[shard, word] |= 1 << bit
            expt_boolmask[shard, e] = True

    if not all(c == 1 for c in counts.values()):
        raise ValueError("each expert must be owned by exactly one shard")

    expt_bitmask = expt_bitmask.to(device)
    expt_boolmask = expt_boolmask.to(device)

    expt_map = torch.full((n_shards, n_expts_tot), -1, dtype=torch.int32)
    for shard, experts in expt_dict.items():
        for local_id, global_id in enumerate(sorted(experts)):
            expt_map[shard, global_id] = local_id
    expt_map = expt_map.to(device)

    n_expts_per_shard = [len(expt_dict[s]) for s in range(n_shards)]
    return ExptAssignment(expt_bitmask, expt_boolmask, expt_map, n_expts_per_shard)


# ---------------------------------------------------------------------------
# Ragged Tensor Metadata
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Top-k Routing / Bitmatrix Metadata
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Masked Reduce
# ---------------------------------------------------------------------------


@triton.jit
def _reduce_kernel(
    Y_ptr,
    stride_y_t,
    stride_y_a,
    stride_y_d,
    Z_ptr,
    stride_z_t,
    stride_z_d,
    Mask_ptr,
    n_tokens,
    d_model,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    if pid_t >= n_tokens:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_model

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for act in range(N_EXPTS_ACT):
        if HAS_MASK:
            m = tl.load(
                Mask_ptr + pid_t * N_EXPTS_ACT * d_model + act * d_model + offs_d,
                mask=mask_d,
                other=0,
            ).to(tl.int1)
        y = tl.load(
            Y_ptr + pid_t * stride_y_t + act * stride_y_a + offs_d * stride_y_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        if HAS_MASK:
            y = tl.where(m, y, 0.0)
        acc += y

    tl.store(
        Z_ptr + pid_t * stride_z_t + offs_d * stride_z_d,
        acc.to(Z_ptr.dtype.element_ty),
        mask=mask_d,
    )


def reduce(
    y: torch.Tensor,
    dim: int = 1,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Sum-reduce over *dim* with optional boolean mask.

    Matches the upstream ``reduce(y, dim=1, mask=mask)`` signature.

    Args:
        y: (n_tokens, k, d_model) expert outputs.
        dim: reduction dimension (must be 1).
        mask: (n_tokens, k, d_model) bool/int mask; zero = skip.

    Returns:
        (z, None) where z has shape (n_tokens, d_model).
    """
    assert dim == 1 and y.ndim == 3
    n_tokens, k, d_model = y.shape
    device = y.device

    z = torch.zeros((n_tokens, d_model), dtype=y.dtype, device=device)

    BLOCK_D = min(triton.next_power_of_2(d_model), 512)
    grid = (n_tokens, triton.cdiv(d_model, BLOCK_D))

    _reduce_kernel[grid](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        z,
        z.stride(0),
        z.stride(1),
        mask if mask is not None else y,
        n_tokens,
        d_model,
        N_EXPTS_ACT=k,
        BLOCK_D=BLOCK_D,
        HAS_MASK=(mask is not None),
    )
    return z, None
