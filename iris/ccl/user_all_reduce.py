# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
User-facing all-reduce that accepts user tensors directly.

Handles copy to/from symmetric heap internally. The caller does not
need to manage heap buffers. Returns a heap-resident result tensor
(stable address for graph capture).
"""

import torch
from iris.ccl.config import Config


class UserAllReduce:
    """All-reduce that takes user tensors and handles heap copy internally.

    Pre-allocates heap buffers per (shape, dtype) for stable addresses
    across CUDA graph capture and replay.
    """

    def __init__(self, ctx, max_numel: int, dtype: torch.dtype = torch.bfloat16):
        self._ctx = ctx
        self._max_numel = max_numel
        self._dtype = dtype
        self._buf_cache = {}
        self._ws_cache = {}
        self._config = Config(
            all_reduce_variant="two_shot",
            block_size_m=32,
            block_size_n=64,
            all_reduce_distribution=1,
        )

    def _get_bufs(self, shape, dtype):
        key = (shape, dtype)
        if key not in self._buf_cache:
            numel = 1
            for s in shape:
                numel *= s
            inp_buf = self._ctx.empty(shape, dtype=dtype)
            out_buf = self._ctx.empty(shape, dtype=dtype)
            self._buf_cache[key] = (inp_buf, out_buf)
        return self._buf_cache[key]

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """All-reduce a user tensor. Returns heap-resident result.

        Args:
            inp: User tensor (any device memory, not necessarily on heap).
                 Must be contiguous and a supported dtype.

        Returns:
            Heap-resident tensor with the all-reduced result.
            Same shape and dtype as input. Stable address across calls
            with the same shape/dtype (safe for graph capture).
        """
        inp_buf, out_buf = self._get_bufs(inp.shape, inp.dtype)
        inp_buf.copy_(inp)

        key = (inp.shape, inp.dtype)
        ws = self._ws_cache.get(key)

        from iris.ccl.all_reduce import all_reduce

        ws = all_reduce(
            out_buf,
            inp_buf,
            self._ctx,
            async_op=True,
            config=self._config,
            workspace=ws,
        )
        self._ws_cache[key] = ws

        return out_buf
