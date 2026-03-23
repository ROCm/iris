################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Ring Attention layer based on:
#   "Ring Attention with Blockwise Transformers for Near-Infinite Context"
#   Liu et al., 2023 (https://arxiv.org/pdf/2310.01889)
#
################################################################################

import torch
import torch.nn as nn

from ring_attention_kernels import ring_attn_fwd


class RingAttention(nn.Module):
    """
    Ring Attention layer for sequence-parallel attention over very long sequences.

    The sequence is assumed to be **already split** across devices along the
    sequence dimension before calling ``forward``.  Each device receives a
    contiguous chunk of Q, K, and V of shape ``[seq_local, num_heads, head_dim]``.

    Internally the layer implements the ring attention algorithm from Liu et al.
    (2023): K and V rotate around the device ring while Q stays local, with
    online softmax accumulation at every step.

    Args:
        shmem: Iris shmem context used for ``barrier()`` and rank queries.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        causal (bool): Whether to apply a causal (lower-triangular) attention
            mask.  Default: ``True``.
        scale (float | None): Softmax scale.  Defaults to
            ``head_dim ** -0.5``.

    Example::

        shmem = iris.iris()
        layer = RingAttention(shmem, num_heads=16, head_dim=64)
        q = torch.randn(seq_local, 16, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output = layer(q, k, v)   # [seq_local, 16, 64]
    """

    def __init__(self, shmem, num_heads: int, head_dim: int, causal: bool = True, scale: float | None = None):
        super().__init__()
        self.shmem = shmem
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        self.scale = scale if scale is not None else head_dim**-0.5
        # Ping-pong buffer cache: keyed by (shape, dtype) to avoid re-allocating
        # the symmetric heap buffers on every forward pass.
        self._buf_cache: dict[
            tuple[torch.Size, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        # Signal flags on the symmetric heap for device-side synchronization.
        # Allocated lazily on first forward call (needs world_size from shmem).
        self._signal_flags: torch.Tensor | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute ring attention.

        Args:
            q: Query tensor ``[seq_local, num_heads, head_dim]``.
            k: Key tensor ``[seq_local, num_heads, head_dim]``.
            v: Value tensor ``[seq_local, num_heads, head_dim]``.

        Returns:
            Attention output tensor ``[seq_local, num_heads, head_dim]``.
        """
        assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
        assert q.shape[1] == self.num_heads, f"Expected {self.num_heads} heads, got {q.shape[1]}"
        assert q.shape[2] == self.head_dim, f"Expected head_dim {self.head_dim}, got {q.shape[2]}"

        # Lazily allocate (or reuse) ping-pong symmetric heap buffers for this shape.
        buf_key = (k.shape, k.dtype)
        if buf_key not in self._buf_cache:
            self._buf_cache[buf_key] = (
                self.shmem.empty(k.shape, dtype=k.dtype),
                self.shmem.empty(k.shape, dtype=k.dtype),
                self.shmem.empty(v.shape, dtype=v.dtype),
                self.shmem.empty(v.shape, dtype=v.dtype),
            )
        ping_pong = self._buf_cache[buf_key]

        # Lazily allocate signal flags on the symmetric heap.
        if self._signal_flags is None:
            world_size = self.shmem.get_num_ranks()
            self._signal_flags = self.shmem.zeros((world_size,), dtype=torch.int32)

        return ring_attn_fwd(
            q, k, v, self.shmem,
            causal=self.causal, scale=self.scale,
            _ping_pong_bufs=ping_pong, _signal_flags=self._signal_flags,
        )
