# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
MoE token dispatch/combine for expert-parallel inference via iris symmetric heap.

Provides ``MoEDispatcher`` — a pre-allocated, handle-based API for routing
tokens to expert-owning ranks (dispatch) and sending results back with
aggregation (combine).

Dispatch uses direct iris.store scatter (not AllToAll/AllToAllv) for sparse,
routing-dependent token movement.  Buffers are allocated once in __init__
and sliced per-call to amortize allocation overhead.

Kernels are the same as examples/31_expert_sharded_moe/{dispatch,combine}.py,
promoted here for production use.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
import iris

from .moe_utils import (
    ExptAssignment,
    RaggedTensorMetadata,
    _make_bitmatrix_metadata,
    make_ragged_tensor_metadata,
    remap_ragged_tensor_metadata,
    reduce,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MoEDispatchConfig:
    """Tuning knobs for MoE dispatch/combine kernels."""

    dispatch_block_size: int = 512  # Tile size for dispatch kernel
    combine_block_size: int = 512  # Tile size for combine kernel


# ---------------------------------------------------------------------------
# DispatchHandle — opaque state passed from dispatch() to combine()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchHandle:
    """Opaque handle returned by ``dispatch()`` and consumed by ``combine()``.

    Carries routing metadata and buffer references so that ``combine()``
    does not need to recompute them.
    """

    expt_assignment: ExptAssignment
    expt_indx_global: torch.Tensor  # (T_global, k) int32
    dispatch_indx: torch.Tensor  # (T_global * k,) row_sorted_indx
    combine_indx: torch.Tensor  # (T_global * k,) col_sorted_indx
    topk_vals: torch.Tensor  # (T_global, k) gating weights
    ragged_meta_global: RaggedTensorMetadata
    expt_sizes: torch.Tensor  # (n_expts,) per-expert counts
    dispatch_buffer: torch.Tensor  # (T_global * k, H) on shmem heap
    n_tokens_local: int
    n_tokens_global: int
    hidden_dim: int
    topk: int


# ---------------------------------------------------------------------------
# Triton kernels (from examples/31_expert_sharded_moe/)
# ---------------------------------------------------------------------------


@triton.jit
def _convert_dp_to_ep(
    dst_ptr,
    dst_stride_m,
    src_ptr,
    src_stride_m,
    src_shape_n,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    expt_indx_stride_m,
    dst_row_indx_ptr,
    dst_row_indx_stride_m,
    n_tokens_local,
    heap_bases,
    SRC_RANK: tl.constexpr,
    N_EXPT_ACT: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_m_global = pid_m + n_tokens_local * SRC_RANK
    off_m_local = pid_m

    offs_n = tl.arange(0, BLOCK)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK), BLOCK)

    for act in tl.static_range(N_EXPT_ACT):
        dst_row = tl.load(dst_row_indx_ptr + off_m_global * dst_row_indx_stride_m + act)
        if dst_row >= 0:
            expt_id = tl.load(expt_indx_ptr + off_m_global * expt_indx_stride_m + act)

            dst_rank = 0
            for r in tl.static_range(N_RANKS):
                word = expt_id // 32
                bit = expt_id % 32
                filt = tl.load(expt_filter_ptr + r * expt_filter_stride_m + word)
                if (filt >> bit) & 1:
                    dst_rank = r

            for start_n in range(0, src_shape_n, BLOCK):
                mask_n = start_n + offs_n < src_shape_n
                src = tl.load(
                    src_ptr + off_m_local * src_stride_m + start_n + offs_n,
                    mask=mask_n,
                    other=0.0,
                )
                dst_off = dst_row * dst_stride_m + start_n + offs_n
                for r in tl.static_range(N_RANKS):
                    if dst_rank == r:
                        iris.store(dst_ptr + dst_off, src, SRC_RANK, r, heap_bases, mask=mask_n, hint=16)


@triton.jit
def _convert_ep_to_dp(
    dst_ptr,
    dst_stride_m,
    src_ptr,
    src_stride_m,
    src_shape_n,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    dst_row_indx_ptr,
    n_slots_per_rank,
    heap_bases,
    BLOCK: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    dst_indx_global = tl.load(dst_row_indx_ptr + pid_m)
    if dst_indx_global < 0:
        return

    dst_rank = dst_indx_global // n_slots_per_rank

    dst_expt_indx = tl.load(expt_indx_ptr + dst_indx_global).to(tl.int32)
    expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
    has_dst_expt = (tl.load(expt_filter_ptr_local + dst_expt_indx // 32) >> (dst_expt_indx % 32)) & 1
    if not has_dst_expt.to(tl.int1):
        return

    dst_indx_local = dst_indx_global - dst_rank * n_slots_per_rank

    offs_n = tl.arange(0, BLOCK)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK), BLOCK)
    for start_n in range(0, src_shape_n, BLOCK):
        mask_n = start_n + offs_n < src_shape_n
        src = tl.load(
            src_ptr + pid_m * src_stride_m + start_n + offs_n,
            mask=mask_n,
            other=0.0,
        )
        dst_off = dst_indx_local * dst_stride_m + start_n + offs_n
        for r in tl.static_range(N_RANKS):
            if dst_rank == r:
                iris.store(dst_ptr + dst_off, src, SRC_RANK, r, heap_bases, mask=mask_n, hint=16)


# ---------------------------------------------------------------------------
# MoEDispatcher
# ---------------------------------------------------------------------------


class MoEDispatcher:
    """Pre-allocated MoE token dispatch/combine for expert-parallel inference.

    Usage::

        dispatcher = ctx.ccl.moe_dispatcher(hidden_dim=4096, num_experts=64,
                                            topk=2, max_tokens=4096)
        # In forward pass:
        recv_tokens, local_meta, handle = dispatcher.dispatch(tokens, topk_idx, topk_weight)
        expert_out = run_experts(recv_tokens, local_meta)  # grouped matmul
        combined = dispatcher.combine(expert_out, handle)
    """

    def __init__(
        self,
        ctx,
        hidden_dim: int,
        num_experts: int,
        topk: int,
        max_tokens: int,
        group=None,
        config: MoEDispatchConfig | None = None,
        expt_assignment: ExptAssignment | None = None,
    ):
        """Pre-allocate buffers for dispatch/combine.

        Args:
            ctx: iris.Iris instance.
            hidden_dim: model hidden dimension (H).
            num_experts: total number of experts across all ranks.
            topk: number of experts activated per token (k).
            max_tokens: maximum tokens per rank per call.
            group: reserved for future process group support.
            config: kernel tuning config.
            expt_assignment: expert-to-rank mapping.  If None, uses uniform
                contiguous assignment.
        """
        self._ctx = ctx
        self._hidden_dim = hidden_dim
        self._num_experts = num_experts
        self._topk = topk
        self._max_tokens = max_tokens
        self._config = config or MoEDispatchConfig()

        self._rank = ctx.get_rank()
        self._world_size = ctx.get_num_ranks()
        self._device = torch.device(f"cuda:{self._rank}")

        # Expert assignment
        if expt_assignment is not None:
            self._expt_assignment = expt_assignment
        else:
            from .moe_utils import make_expt_dict_uniform, make_expt_assignment

            expt_dict = make_expt_dict_uniform(self._world_size, num_experts)
            self._expt_assignment = make_expt_assignment(self._world_size, num_experts, expt_dict, self._device)

        # Pre-allocate buffers
        max_T_global = max_tokens * self._world_size
        max_slots = max_T_global * topk

        self._dispatch_buf = ctx.zeros((max_slots, hidden_dim), dtype=torch.bfloat16)
        self._combine_buf = ctx.zeros((max_tokens * topk, hidden_dim), dtype=torch.bfloat16)
        self._ag_indx_buf = ctx.zeros((max_T_global, topk), dtype=torch.int32)
        self._ag_vals_buf = ctx.zeros((max_T_global, topk), dtype=torch.float32)

    def dispatch(
        self,
        tokens: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, RaggedTensorMetadata, DispatchHandle]:
        """Dispatch local tokens to expert-owning ranks.

        Args:
            tokens: (T_local, H) local token activations.
            topk_idx: (T_local, k) int expert indices from gating.
            topk_weight: (T_local, k) float gating weights.

        Returns:
            (dispatch_buffer, local_ragged_meta, handle) where:
            - dispatch_buffer: (n_active_slots, H) tokens in expert-sorted order
              on this rank's symmetric heap.
            - local_ragged_meta: RaggedTensorMetadata for this rank's experts.
            - handle: DispatchHandle to pass to combine().
        """
        ctx = self._ctx
        rank = self._rank
        world_size = self._world_size
        k = self._topk
        hidden_dim = self._hidden_dim
        config = self._config

        n_tokens_local = tokens.shape[0]
        n_tokens_global = n_tokens_local * world_size

        # Step 1: Promote indices to int32 (narrow type corruption bug)
        topk_idx_i32 = topk_idx.contiguous().to(torch.int32)
        topk_weight_f32 = topk_weight.contiguous().to(torch.float32)

        # Step 2: All-gather topk_idx and topk_weight via ctx.ccl.all_gather
        ag_indx = self._ag_indx_buf[:n_tokens_global, :k]
        ag_vals = self._ag_vals_buf[:n_tokens_global, :k]
        ctx.ccl.all_gather(ag_indx, topk_idx_i32)
        ctx.ccl.all_gather(ag_vals, topk_weight_f32)

        indx_global = ag_indx  # (T_global, k) int32
        vals_global = ag_vals  # (T_global, k) float32

        # Step 3: Build BitmatrixMetadata from global indices
        mask_metadata = _make_bitmatrix_metadata(indx_global.to(torch.int32), self._num_experts)
        dispatch_indx = mask_metadata.row_sorted_indx  # (T_global * k,)
        combine_indx = mask_metadata.col_sorted_indx  # (T_global * k,)
        expt_sizes = mask_metadata.col_sum  # (n_expts,)

        # Step 4: Build RaggedTensorMetadata
        n_active = int(expt_sizes.sum().item())
        ragged_meta_global = make_ragged_tensor_metadata(expt_sizes, n_active)

        # Step 5: Zero dispatch buffer, barrier
        n_total_slots = n_tokens_global * k
        dispatch_buf = self._dispatch_buf[:n_total_slots, :hidden_dim]
        dispatch_buf.zero_()
        ctx.barrier()

        # Step 6: Launch _convert_dp_to_ep kernel
        BLOCK = min(triton.next_power_of_2(hidden_dim), config.dispatch_block_size)
        grid = (n_tokens_local,)

        expt_bitmask = self._expt_assignment.expt_bitmask

        _convert_dp_to_ep[grid](
            dispatch_buf,
            dispatch_buf.stride(0),
            tokens,
            tokens.stride(0),
            tokens.shape[1],
            expt_bitmask,
            expt_bitmask.stride(0),
            indx_global,
            indx_global.stride(0),
            dispatch_indx,
            k,
            n_tokens_local,
            ctx.get_heap_bases(),
            SRC_RANK=rank,
            N_EXPT_ACT=k,
            N_RANKS=world_size,
            BLOCK=BLOCK,
        )

        # Step 7: Barrier (all stores must complete before reads)
        ctx.barrier()

        # Step 8: Remap ragged metadata to local expert view
        expt_map = self._expt_assignment.expt_map[rank, :].contiguous()
        local_ragged_meta = remap_ragged_tensor_metadata(ragged_meta_global, expt_map)

        # Build handle
        handle = DispatchHandle(
            expt_assignment=self._expt_assignment,
            expt_indx_global=indx_global,
            dispatch_indx=dispatch_indx,
            combine_indx=combine_indx,
            topk_vals=vals_global,
            ragged_meta_global=ragged_meta_global,
            expt_sizes=expt_sizes,
            dispatch_buffer=dispatch_buf,
            n_tokens_local=n_tokens_local,
            n_tokens_global=n_tokens_global,
            hidden_dim=hidden_dim,
            topk=k,
        )

        return dispatch_buf, local_ragged_meta, handle

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: DispatchHandle,
    ) -> torch.Tensor:
        """Combine expert results back to token-owning ranks.

        Args:
            expert_output: (n_total_slots, H) expert-sorted matmul output.
                These are the results after the grouped expert computation.
            handle: DispatchHandle from dispatch().

        Returns:
            combined: (T_local, H) combined output for this rank's tokens.
        """
        ctx = self._ctx
        rank = self._rank
        world_size = self._world_size
        config = self._config

        n_tokens_local = handle.n_tokens_local
        n_tokens_global = handle.n_tokens_global
        hidden_dim = handle.hidden_dim
        k = handle.topk

        expt_bitmask = handle.expt_assignment.expt_bitmask
        flat_expt_indx = handle.expt_indx_global.to(torch.int32).reshape(-1)
        combine_indx = handle.combine_indx

        # Step 1: Zero combine buffer, barrier
        n_local_slots = n_tokens_local * k
        combine_buf = self._combine_buf[:n_local_slots, :hidden_dim]
        combine_buf.zero_()
        ctx.barrier()

        # Step 2: Launch _convert_ep_to_dp kernel
        # n_slots_per_rank for the combine kernel: n_tokens_local * k
        # because the flat dispatch_indx has n_tokens_global * k entries,
        # and each rank's portion is n_tokens_local * k
        n_slots_per_rank = n_tokens_local * k
        n_total_slots = n_tokens_global * k

        BLOCK = min(triton.next_power_of_2(hidden_dim), config.combine_block_size)
        grid = (n_total_slots,)

        _convert_ep_to_dp[grid](
            combine_buf,
            combine_buf.stride(0),
            expert_output,
            expert_output.stride(0),
            expert_output.shape[1],
            expt_bitmask,
            expt_bitmask.stride(0),
            flat_expt_indx,
            combine_indx,
            n_slots_per_rank,
            ctx.get_heap_bases(),
            BLOCK=BLOCK,
            SRC_RANK=rank,
            N_RANKS=world_size,
        )

        # Step 3: Barrier
        ctx.barrier()

        # Step 4: Reshape combine buffer to (T_local, k, H)
        combine_3d = combine_buf.view(n_tokens_local, k, hidden_dim)

        # Step 5: Masked reduce over dim=1
        dispatch_indx = handle.dispatch_indx
        y_mask = (dispatch_indx != -1).view(n_tokens_global, k, 1)
        local_mask = y_mask[rank * n_tokens_local : (rank + 1) * n_tokens_local]
        local_mask = local_mask.expand_as(combine_3d).contiguous()
        combined, _ = reduce(combine_3d, dim=1, mask=local_mask)

        return combined
