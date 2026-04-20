#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for distributed flash decode."""

import sys
from pathlib import Path

import torch
import iris.bench as bench

# Add the flash decode example to the path.
_project_root = Path(__file__).resolve()
while not (_project_root / "tests").is_dir() or not (_project_root / "examples").is_dir():
    if _project_root == _project_root.parent:
        raise FileNotFoundError("Could not find project root")
    _project_root = _project_root.parent

_module_dir = _project_root / "examples" / "13_flash_decode"
if _module_dir.is_dir():
    sys.path.insert(0, str(_module_dir))
else:
    raise FileNotFoundError(f"Target directory not found: {_module_dir}")

from flash_decode_fused_layer import flash_decode_fused_layer  # noqa: E402


@bench.register
@bench.axis("kv_len", [1024, 4096, 16384, 65536])
@bench.axis("num_heads", [32])
@bench.axis("head_dim", [128])
@bench.axis("num_seqs", [1, 4])
def flash_decode(state, ctx):
    kv_len = state["kv_len"]
    num_heads = state["num_heads"]
    head_dim = state["head_dim"]
    num_seqs = state["num_seqs"]
    dtype = torch.float16

    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    num_kv_heads = num_heads // 8 if num_heads >= 8 else 1
    scale = head_dim**-0.5
    block_size = 1

    fd_layer = flash_decode_fused_layer(
        ctx,
        rank,
        rank,
        world_size,
        world_size,
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        q_head_dim=head_dim,
        v_head_dim=head_dim,
        page_size=block_size,
        scale=scale,
        soft_cap=0.0,
        max_allowed_batch=num_seqs,
    )

    device = torch.device(f"cuda:{rank}")

    num_blocks = (kv_len + block_size - 1) // block_size
    query = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(num_seqs, 1)

    kv_lens_per_rank = [kv_len] * num_seqs
    kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=device)
    global_kv_lens = kv_lens_tensor.unsqueeze(0).repeat(world_size, 1)

    state.add_counter("global_kv_len", float(kv_len * world_size))

    clear_fn = getattr(fd_layer, "clear_flags", None)

    state.exec(
        lambda: fd_layer(query, key_cache, value_cache, global_kv_lens, block_tables),
        preamble_fn=clear_fn,
    )


if __name__ == "__main__":
    bench.main()
