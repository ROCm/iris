################################################################################
# Part of the code adapted from
# https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/layers/nvidia/sp_flash_decode_layer.py################################################################################
################################################################################

import torch
import triton
import math
import iris

from decode_kernels import (
    gqa_local_kernels_fused,
    gqa_global_reduce_fused
)

class FDFusedLayer(torch.nn.Module):
    def __init__(self, iris_instance, rank, node, num_ranks, num_nodes, num_q_heads, num_kv_heads, q_head_dim, v_head_dim, page_size=1,
                 scale=1, soft_cap=0, max_allowed_batch=1, thrink_buffer_threshold=500, stages=20):
        super().__init__()
        self.iris_instance = iris_instance
        self.rank = rank
        self.num_ranks = num_ranks
        self.node = node
        self.num_nodes = num_nodes

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.soft_cap = soft_cap
        self.scale = scale
        self.kv_split = 32 
        self.max_allowed_batch = max_allowed_batch
        
        self.BLOCK_DV = triton.next_power_of_2(self.v_head_dim)

        self.gathered_buffer = self.iris_instance.empty(
            (self.num_ranks, self.max_allowed_batch, self.num_q_heads, self.v_head_dim + 1),
            dtype=torch.float16
        )
        
        # Use per-tile signaling for finer-grained synchronization
        # This will tell which rank sent the data to which rank, for each batch item and head
        self.signal_flags = self.iris_instance.zeros(
            (self.num_ranks, self.num_ranks, self.max_allowed_batch, self.num_q_heads), dtype=torch.int32
        )
        
        # self.producer_stream = torch.cuda.Stream()
        # self.consumer_stream = torch.cuda.Stream()
        

    def clear_flags(self):
        """Resets synchronization flags for the next iteration."""
        self.signal_flags.zero_()
        self.iris_instance.barrier()

    def forward(self, q, k_cache, v_cache, global_kv_lens, block_table):
        batch = q.shape[0]
        assert global_kv_lens.shape[0] == self.num_ranks
        assert global_kv_lens.shape[1] == batch
        assert batch <= self.max_allowed_batch
    
        output_split = torch.empty(
            [batch, self.num_q_heads, self.kv_split, self.v_head_dim + 1], dtype=q.dtype, device=q.device
        )
        final_output = torch.empty([batch, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device)
        
        
        # with torch.cuda.stream(self.producer_stream):
        gqa_local_kernels_fused(
            q, k_cache, v_cache,
            self.gathered_buffer, self.signal_flags, self.iris_instance,
            [1] * batch, global_kv_lens[self.rank], block_table, self.scale,
            soft_cap=self.soft_cap, output_split=output_split, kv_split=self.kv_split
        )
    
        
        # with torch.cuda.stream(self.consumer_stream):
        kk3 = gqa_global_reduce_fused[(batch, self.num_q_heads)](
            self.gathered_buffer,
            final_output,
            global_kv_lens,
            self.signal_flags,
            self.signal_flags.stride(0),    # stride_signal_dest
            self.signal_flags.stride(1),    # stride_signal_src
            self.signal_flags.stride(2),    # stride_signal_bs
            self.signal_flags.stride(3),    # stride_signal_h
            batch,
            self.num_q_heads,
            self.gathered_buffer.stride(1), # stride_mid_ob
            self.gathered_buffer.stride(2), # stride_mid_oh
            self.gathered_buffer.stride(0), # stride_mid_os (now rank stride)
            final_output.stride(0),         # stride_obs
            final_output.stride(1),         # stride_oh
            self.rank,
            self.num_ranks,                 # NUM_KV_SPLITS becomes num_ranks
            self.BLOCK_DV,
            self.v_head_dim,
        )

        # print(f"{kk3.n_regs} registers used third, {kk3.n_spills} spills")

        # self.clear_flags()
        
        return final_output