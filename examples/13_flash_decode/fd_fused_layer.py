import torch
import triton
import math
import iris

from decode_kernels import (
    gqa_local_decode_split_k,
    gqa_local_reduce_fused_full,
    gqa_global_reduce_fused_full
)

def gqa_local_kernels_fused_full(
    q, k_cache, v_cache,
    
    gathered_buffer, signal_flags, iris_instance,

    q_lens, kv_lens, block_table, scale, soft_cap=0.0,
    output_split=None, kv_split=-1,
):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    v_head_dim = v_cache.shape[-1]
    rank = iris_instance.get_rank()
    num_ranks = iris_instance.get_num_ranks()

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)
    kv_group_num = q_heads // kv_heads
    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    # Step 1: Split-K calculation (same as before)
    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)
    if output_split is None:
        output_split = torch.empty(
            [batch, q_heads, NUM_KV_SPLITS, v_head_dim + 1], dtype=q.dtype, device=q.device
        )

    gqa_local_decode_split_k[grid_split_kv](
        q, k_cache, v_cache, output_split, scale, block_table, kv_lens,
        batch, q.stride(0), q.stride(1),
        k_cache.stride(-3), k_cache.stride(-2), v_cache.stride(-3), v_cache.stride(-2),
        output_split.stride(0), output_split.stride(1), output_split.stride(2),
        block_table.stride(0), kv_group_num, q_heads,
        BLOCK_HEAD_DIM, BLOCK_DPE, BLOCK_DV, BLOCK_N, BLOCK_H, NUM_KV_SPLITS,
        page_size, soft_cap, k_head_dim, v_head_dim,
        num_warps=4, num_stages=2
    )

    # Step 2: Fused Intra-Rank Combine and Inter-Rank Push with tile-level signaling
    grid_combine_push = (batch, q_heads)
    gqa_local_reduce_fused_full[grid_combine_push](
        output_split,
        kv_lens,
        gathered_buffer,
        signal_flags,
        signal_flags.stride(0), signal_flags.stride(1), signal_flags.stride(2), signal_flags.stride(3),
        iris_instance.get_heap_bases(),
        output_split.stride(0), output_split.stride(1), output_split.stride(2),
        gathered_buffer.stride(0), gathered_buffer.stride(1), gathered_buffer.stride(2),
        rank,
        num_ranks,
        q_heads,
        NUM_KV_SPLITS,
        BLOCK_DV,
        v_head_dim,
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
        gqa_local_kernels_fused_full(
            q, k_cache, v_cache,
            self.gathered_buffer, self.signal_flags, self.iris_instance,
            [1] * batch, global_kv_lens[self.rank], block_table, self.scale,
            soft_cap=self.soft_cap, output_split=output_split, kv_split=self.kv_split
        )
    
        
        # with torch.cuda.stream(self.consumer_stream):
        kk3 = gqa_global_reduce_fused_full[(batch, self.num_q_heads)](
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