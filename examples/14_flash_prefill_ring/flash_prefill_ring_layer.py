# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Production-ready PyTorch module for fused ring prefill attention.

Mirrors the design of examples/13_flash_decode/flash_decode_fused_layer.py:
- Pre-allocates all symmetric-heap buffers at init time
- Exposes a clean forward() API
- Supports flag clearing between forward passes via clear_flags()

Phase 9: flash_prefill_ring_layer
"""

import sys
from pathlib import Path

import torch
import triton

# Add iris.x to path
project_root = Path(__file__).resolve()
while not (project_root / "iris").is_dir():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import iris
import iris.x


class flash_prefill_ring_layer(torch.nn.Module):
    """
    Fused ring prefill attention layer.

    Implements distributed multi-head (GQA-compatible) prefill attention
    across `num_ranks` GPUs using a ring communication pattern pipelined
    with attention compute in a single Triton kernel.

    The ring eliminates the need for a full AllGather of KV, reducing peak
    memory and overlapping compute with communication.

    Args:
        shmem: iris.Iris symmetric heap instance
        rank: current rank (0-indexed)
        num_ranks: total number of ranks (world_size)
        num_q_heads: number of query heads
        num_kv_heads: number of key/value heads (GQA: num_q_heads >= num_kv_heads)
        head_dim: dimension per head
        block_size: KV cache page size (tokens per block)
        scale: attention scale (default: head_dim ** -0.5)
        max_chunk_len: maximum tokens per rank per forward pass
    """

    def __init__(
        self,
        shmem,
        rank: int,
        num_ranks: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        scale: float = None,
        max_chunk_len: int = 2048,
    ):
        super().__init__()
        self.shmem = shmem
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.max_chunk_len = max_chunk_len

        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads={num_q_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
        self.H_PER_KV = num_q_heads // num_kv_heads

        # Pre-compute HEAD_DIM (power of 2 for Triton)
        self.HEAD_DIM = triton.next_power_of_2(head_dim)

        # Size of one KV chunk: max_chunk_len tokens × num_kv_heads × HEAD_DIM
        chunk_kv_elements = max_chunk_len * num_kv_heads * self.HEAD_DIM

        # Pre-allocate double ring buffers on symmetric heap
        # Each buffer holds one rank's full KV chunk
        self.k_ring_A = shmem.empty(chunk_kv_elements, dtype=torch.float16)
        self.v_ring_A = shmem.empty(chunk_kv_elements, dtype=torch.float16)
        self.k_ring_B = shmem.empty(chunk_kv_elements, dtype=torch.float16)
        self.v_ring_B = shmem.empty(chunk_kv_elements, dtype=torch.float16)

        # Signal flags: one int32 per KV head per buffer (A and B)
        # Layout: [num_kv_heads] — each head has its own flag
        self.signal_flags_A = shmem.zeros((num_kv_heads,), dtype=torch.int32)
        self.signal_flags_B = shmem.zeros((num_kv_heads,), dtype=torch.int32)

        # Device context tensor for kernel initialization
        self.context_tensor = shmem.get_device_context()

    def clear_flags(self):
        """
        Reset synchronization flags between forward passes.

        Must be called before each forward pass (or can be done after).
        Includes a barrier to ensure all ranks reset before proceeding.
        """
        self.signal_flags_A.zero_()
        self.signal_flags_B.zero_()
        self.shmem.barrier()

    def forward(
        self,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        chunk_len: int = None,
    ) -> torch.Tensor:
        """
        Run fused ring prefill attention.

        Args:
            q: [S_local, num_q_heads, head_dim] query tensor (fp16, on this rank)
            k_local: [S_local, num_kv_heads, head_dim] key tensor (fp16, sym heap)
            v_local: [S_local, num_kv_heads, head_dim] value tensor (fp16, sym heap)
            chunk_len: sequence length per rank (default: q.shape[0])

        Returns:
            out: [S_local, num_q_heads, head_dim] output tensor (fp16)
        """
        S_local, H, D = q.shape
        H_kv = k_local.shape[1]

        if chunk_len is None:
            chunk_len = S_local

        assert S_local <= self.max_chunk_len, (
            f"S_local={S_local} exceeds max_chunk_len={self.max_chunk_len}"
        )
        assert H == self.num_q_heads
        assert H_kv == self.num_kv_heads
        assert D == self.head_dim

        HEAD_DIM = self.HEAD_DIM

        # Pad head dim to power of 2 if needed
        if HEAD_DIM != D:
            q = torch.nn.functional.pad(q, (0, HEAD_DIM - D))
            k_local = torch.nn.functional.pad(k_local, (0, HEAD_DIM - D))
            v_local = torch.nn.functional.pad(v_local, (0, HEAD_DIM - D))

        out = torch.empty(S_local, H, HEAD_DIM, dtype=torch.float16, device=q.device)

        BLOCK_Q = 64
        BLOCK_K = 64

        # Grid: (1, H) — one CTA per query head, iterating Q-tiles internally.
        # This ensures exactly one CTA per kv_h handles the ring flags (no race).
        grid = (1, H)

        iris.x.fused_ring_prefill_attn_kernel[grid](
            q,
            k_local, v_local,
            self.k_ring_A, self.v_ring_A,
            self.k_ring_B, self.v_ring_B,
            self.signal_flags_A,
            self.signal_flags_B,
            out,
            self.context_tensor,
            # Q strides
            q.stride(0), q.stride(1), q.stride(2),
            # Local KV strides
            k_local.stride(0), k_local.stride(1), k_local.stride(2),
            # Ring buffer strides (flat layout: chunk_len * H_kv * HEAD_DIM)
            H_kv * HEAD_DIM, HEAD_DIM, 1,
            # Output strides
            out.stride(0), out.stride(1), out.stride(2),
            # metadata
            S_local, H, H_kv,
            chunk_len,
            self.scale,
            cur_rank=self.rank,
            world_size=self.num_ranks,
            H_PER_KV=self.H_PER_KV,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
        )

        if HEAD_DIM != D:
            out = out[:, :, :D]

        return out.contiguous()
