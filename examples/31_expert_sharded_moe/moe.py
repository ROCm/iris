# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Expert-sharded MoE forward pass -- reference and distributed.

Closely follows the test flow in triton_kernels/tests/test_distributed.py:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_distributed.py

Implements:
  mixture_of_expt_nosharded  -- single-device reference (uses PyTorch)
  mixture_of_expt_epsharded  -- expert-parallel 8-step pipeline using iris
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
import iris

from ragged_metadata import make_ragged_tensor_metadata, remap_ragged_tensor_metadata
from topk import topk, _make_bitmatrix_metadata
from dispatch import convert_dp_to_ep
from combine import convert_ep_to_dp
from grouped_matmul import grouped_matmul
from fused_exp_matmul_ep_to_dp import fused_exp_matmul_ep_to_dp
from reduce import reduce


# ---------------------------------------------------------------------------
# Iris all-gather helper (push model)
# ---------------------------------------------------------------------------


@triton.jit
def _allgather_push_kernel(
    src_ptr,
    dst_ptr,
    dst_offset,
    src_numel,
    heap_bases,
    CUR_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < src_numel
    data = tl.load(src_ptr + offs, mask=mask)
    for r in tl.static_range(N_RANKS):
        dst = dst_ptr + dst_offset + offs
        iris.store(dst, data, CUR_RANK, r, heap_bases, mask=mask, hint=16)


def _allgather_iris(local_tensor, shmem):
    """All-gather a 2-D tensor via iris push: each rank writes its chunk
    to every rank's shared buffer at the correct offset.

    Sub-32-bit dtypes (e.g. int16) are promoted to int32 for the push
    because iris.store can silently mishandle narrow element types when
    the heap offset is not aligned to the natural store width.
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    orig_dtype = local_tensor.dtype
    need_promote = orig_dtype.itemsize < 4
    work_dtype = torch.int32 if need_promote else orig_dtype

    src = local_tensor.contiguous()
    if need_promote:
        src = src.to(work_dtype)

    n_local = src.shape[0]
    rest = list(src.shape[1:])
    global_shape = [n_local * world_size] + rest
    buf = shmem.zeros(global_shape, dtype=work_dtype)
    # Match the other communication wrappers: ensure every rank has
    # allocated its destination heap buffer before remote stores begin.
    shmem.barrier()
    heap_bases = shmem.get_heap_bases()

    src_flat = src.view(-1)
    numel = src_flat.numel()
    elem_offset = rank * numel

    BLOCK = min(triton.next_power_of_2(numel), 1024)
    grid = (triton.cdiv(numel, BLOCK),)
    _allgather_push_kernel[grid](
        src_flat,
        buf.view(-1),
        elem_offset,
        numel,
        heap_bases,
        CUR_RANK=rank,
        N_RANKS=world_size,
        BLOCK=BLOCK,
    )
    shmem.barrier()
    if need_promote:
        return buf.to(orig_dtype)
    return buf


# ---------------------------------------------------------------------------
# Reference: single-device MoE (no sharding)
# ---------------------------------------------------------------------------


def mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act):
    """Reference MoE on a single device using our own kernels.

    Follows the upstream routing -> matmul -> reduce flow exactly:
      1. topk routing
      2. build bitmatrix & ragged metadata
      3. gather tokens into expert-sorted order (using col_sorted_indx / k)
      4. grouped matmul
      5. scatter results back to (n_tokens*k, d_model)
      6. reduce(dim=1) with validity mask
    """
    n_tokens, d_model = x_global.shape
    n_expts_tot = l_global.shape[1]
    device = x_global.device

    topk_result = topk(l_global, n_expts_act, apply_softmax=True)
    active_indx = topk_result.indx
    mask_metadata = _make_bitmatrix_metadata(active_indx.to(torch.int32), n_expts_tot)

    dispatch_indx = mask_metadata.row_sorted_indx
    combine_indx = mask_metadata.col_sorted_indx
    expt_sizes = mask_metadata.col_sum

    n_active = int(expt_sizes.sum().item())
    ragged_meta = make_ragged_tensor_metadata(expt_sizes, n_active)

    gather_idx = torch.div(combine_indx[:n_active], n_expts_act, rounding_mode="trunc")

    x_sorted = torch.zeros(n_active, d_model, dtype=x_global.dtype, device=device)
    valid_gather = gather_idx >= 0
    x_sorted[valid_gather] = x_global[gather_idx[valid_gather].long()]

    y_sorted = grouped_matmul(x_sorted, w_global, b_global, ragged_meta)

    y_flat = torch.zeros(n_tokens * n_expts_act, d_model, dtype=x_global.dtype, device=device)
    for i in range(n_active):
        dst = combine_indx[i].item()
        if dst >= 0:
            y_flat[dst] = y_sorted[i]

    y_mask = (dispatch_indx != -1).view(n_tokens, n_expts_act, 1)
    y_3d = y_flat.view(n_tokens, n_expts_act, d_model)
    y_mask = y_mask.expand_as(y_3d).contiguous()
    y_global, _ = reduce(y_3d, dim=1, mask=y_mask)
    return y_global


# ---------------------------------------------------------------------------
# Distributed: expert-parallel MoE using iris
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeFusionConfig:
    """Fusion mode selector for expert-sharded MoE pipeline."""

    fuse_convert_dp_to_ep_grouped_matmul: bool = False
    fuse_grouped_matmul_convert_ep_to_dp: bool = False

    def mode_name(self) -> str:
        parts: list[str] = []
        if self.fuse_convert_dp_to_ep_grouped_matmul:
            parts.append("convert_dp_to_ep_grouped_matmul")
        if self.fuse_grouped_matmul_convert_ep_to_dp:
            parts.append("grouped_matmul_convert_ep_to_dp")
        if not parts:
            return "unfused"
        return "fused_" + "__".join(parts)

    @staticmethod
    def from_mode_name(name: str) -> "MoeFusionConfig":
        if name == "unfused":
            return MoeFusionConfig()
        if name == "fused_grouped_matmul_convert_ep_to_dp":
            return MoeFusionConfig(fuse_grouped_matmul_convert_ep_to_dp=True)
        if name == "fused_convert_dp_to_ep_grouped_matmul":
            return MoeFusionConfig(fuse_convert_dp_to_ep_grouped_matmul=True)
        if name == "fused_convert_dp_to_ep_grouped_matmul__grouped_matmul_convert_ep_to_dp":
            return MoeFusionConfig(
                fuse_convert_dp_to_ep_grouped_matmul=True,
                fuse_grouped_matmul_convert_ep_to_dp=True,
            )
        raise ValueError(f"Unknown fusion mode name: {name}")


def mixture_of_expt_epsharded(
    x_dp_local,
    l_dp_local,
    w_ep_local,
    b_ep_local,
    expt_assignment,
    n_expts_act,
    shmem,
    fusion_config: MoeFusionConfig | None = None,
    timing_dict: dict | None = None,
):
    """Expert-parallel MoE forward using iris symmetric heap.

    Args:
        x_dp_local: (n_tokens_local, d_model) local token activations.
        l_dp_local: (n_tokens_local, n_expts_tot) local logits.
        w_ep_local: (n_expts_local, d_model, d_model) local expert weights.
        b_ep_local: (n_expts_local, d_model) local expert biases.
        expt_assignment: ExptAssignment mapping experts to ranks.
        n_expts_act: k (experts per token).
        shmem: iris.Iris instance.

    Returns:
        (n_tokens_local, d_model) output for this rank's tokens.
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    n_tokens_local, d_model = x_dp_local.shape
    n_tokens_global = n_tokens_local * world_size
    n_expts_tot = l_dp_local.shape[1]
    device = x_dp_local.device

    def _tick(label):
        """Record a cuda event for timing breakdown. timing_dict is a list of (label, event) pairs."""
        if timing_dict is not None:
            torch.cuda.synchronize()
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            timing_dict.append((label, ev))

    _tick("start")

    # ------------------------------------------------------------------
    # Step 1: Top-k routing (local) + all-gather via iris
    # ------------------------------------------------------------------
    local_topk = topk(l_dp_local, n_expts_act, apply_softmax=True)
    _tick("topk")

    vals_global = _allgather_iris(local_topk.vals, shmem)
    # Keep routing indices in int32 after gather. We observed rank-dependent
    # corruption when converting gathered index buffers back to int16.
    indx_global = _allgather_iris(local_topk.indx.contiguous().to(torch.int32), shmem)
    _tick("allgather")

    # ------------------------------------------------------------------
    # Step 2: Extract routing metadata from global topk
    # ------------------------------------------------------------------
    active_indx = indx_global
    mask_metadata = _make_bitmatrix_metadata(active_indx.to(torch.int32), n_expts_tot)

    expt_sizes = mask_metadata.col_sum
    dispatch_indx = mask_metadata.row_sorted_indx
    combine_indx = mask_metadata.col_sorted_indx

    # ------------------------------------------------------------------
    # Step 3: Build ragged tensor metadata
    # ------------------------------------------------------------------
    n_active = int(expt_sizes.sum().item())
    x_global_metadata = make_ragged_tensor_metadata(expt_sizes, n_active)
    _tick("metadata")

    # ------------------------------------------------------------------
    # Step 4: DP -> EP dispatch (all-to-all via iris.store)
    # ------------------------------------------------------------------
    y_ep_local = convert_dp_to_ep(
        x_dp_local,
        expt_assignment,
        active_indx,
        dispatch_indx,
        shmem,
    )
    _tick("dispatch")

    # ------------------------------------------------------------------
    # Step 5: Remap ragged metadata to local expert view
    # ------------------------------------------------------------------
    expt_map = expt_assignment.expt_map[rank, :].contiguous()
    y_ep_local_metadata = remap_ragged_tensor_metadata(x_global_metadata, expt_map)

    fusion_config = fusion_config or MoeFusionConfig()
    if fusion_config.fuse_convert_dp_to_ep_grouped_matmul:
        raise NotImplementedError("Fusion mode convert_dp_to_ep_grouped_matmul is not implemented yet.")

    # ------------------------------------------------------------------
    # grouped_matmul + convert_ep_to_dp (select fused/unfused variant)
    # ------------------------------------------------------------------
    flat_expt_indx = active_indx.to(torch.int32).reshape(-1)
    if fusion_config.fuse_grouped_matmul_convert_ep_to_dp:
        y_dp_local = fused_exp_matmul_ep_to_dp(
            y_ep_local,
            w_ep_local,
            b_ep_local,
            expt_assignment,
            expt_map,
            flat_expt_indx,
            combine_indx,
            shmem,
            ragged_metadata=y_ep_local_metadata,
        )
        _tick("fused_matmul_scatter")
    else:
        y_ep_local = grouped_matmul(y_ep_local, w_ep_local, b_ep_local, y_ep_local_metadata)
        _tick("matmul")
        y_dp_local = convert_ep_to_dp(
            y_ep_local,
            expt_assignment,
            flat_expt_indx,
            combine_indx,
            shmem,
        )
        _tick("combine")

    # ------------------------------------------------------------------
    # Step 8: Reduce (unweighted sum, masked)
    # ------------------------------------------------------------------
    y_dp_local = y_dp_local.view(-1, n_expts_act, d_model)
    y_mask = (dispatch_indx != -1).view(n_tokens_global, n_expts_act, 1)
    local_mask = y_mask[rank * n_tokens_local : (rank + 1) * n_tokens_local]
    local_mask = local_mask.expand_as(y_dp_local).contiguous()
    z_dp_local, _ = reduce(y_dp_local, dim=1, mask=local_mask)
    _tick("reduce")

    torch.cuda.synchronize()
    return z_dp_local
