################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Part of the code adapted from
# https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/layers/nvidia/sp_flash_decode_layer.py################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


import torch
import torch.distributed as dist
from decode_kernels import gqa_local_kernels, gqa_reduce_global


class flash_decode_layer_rccl(torch.nn.Module):
    def __init__(
        self,
        rank: int,
        num_ranks: int,
        num_q_heads: int,
        num_kv_heads: int,
        q_head_dim: int,
        v_head_dim: int,
        process_group,
        page_size: int = 1,
        scale: float = 1.0,
        soft_cap: float = 0.0,
        max_allowed_batch: int = 1,
    ):
        super().__init__()
        self.rank = rank
        self.num_ranks = num_ranks
        self.process_group = process_group

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.soft_cap = soft_cap
        self.scale = scale

        self.kv_split = 32
        self.max_allowed_batch = max_allowed_batch

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        global_kv_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor, identical across all ranks. Shape: [batch, num_q_heads, head_size]
            k_cache: This rank's shard of the key cache.
            v_cache: This rank's shard of the value cache.
            global_kv_lens: A tensor containing the sequence lengths of the K/V cache shards on all ranks. Shape: [num_ranks, batch]
            block_table: The block table for this rank's K/V cache.
        Returns:
            The final attention output tensor. Shape: [batch, num_q_heads, head_size]
        """
        batch_size = q.shape[0]

        assert global_kv_lens.shape[0] == self.num_ranks, "global_kv_lens must have a dimension for each rank."
        assert global_kv_lens.shape[1] == batch_size, "global_kv_lens batch dimension mismatch."
        assert batch_size <= self.max_allowed_batch, (
            f"Input batch size {batch_size} exceeds max allowed {self.max_allowed_batch}."
        )

        output_combine = torch.empty(
            [batch_size, self.num_q_heads, self.v_head_dim + 1], dtype=q.dtype, device=q.device
        )
        final_output = torch.empty([batch_size, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device)

        all_ranks_output_combine = torch.empty(
            [self.num_ranks, batch_size, self.num_q_heads, self.v_head_dim + 1], dtype=q.dtype, device=q.device
        )

        gqa_local_kernels(
            q,
            k_cache,
            v_cache,
            workspace=None,
            q_lens=[1] * batch_size,
            kv_lens=global_kv_lens[self.rank],
            block_table=block_table,
            scale=self.scale,
            soft_cap=self.soft_cap,
            output_combine=output_combine,
            kv_split=self.kv_split,
        )

        dist.all_gather_into_tensor(all_ranks_output_combine, output_combine, group=self.process_group)

        gqa_reduce_global[(batch_size, self.num_q_heads, 1)](
            all_ranks_output_combine,
            final_output,
            global_kv_lens,
            batch_size,
            self.num_q_heads,
            all_ranks_output_combine.stride(1),  # stride_mid_ob
            all_ranks_output_combine.stride(2),  # stride_mid_oh
            all_ranks_output_combine.stride(0),  # stride_mid_os
            final_output.stride(0),  # stride_obs
            final_output.stride(1),  # stride_oh
            self.num_ranks,  # NUM_KV_SPLITS
            512,  # BLOCK_DV
            self.v_head_dim,  # Lv
        )

        return final_output
