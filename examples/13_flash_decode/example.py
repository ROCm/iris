#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: Flash Decode Fused Attention

A distributed Flash Decode kernel for accelerating LLM inference. The KV cache
is sharded across all ranks; each rank computes local attention scores and
participates in a fused global reduce to produce the final output.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import iris

# The flash_decode_fused_layer module lives alongside this file
sys.path.insert(0, str(Path(__file__).parent))
from flash_decode_fused_layer import flash_decode_fused_layer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flash Decode fused attention example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kv_len_per_rank", type=int, default=32768, help="KV sequence length per rank")
    parser.add_argument("--num_heads", type=int, default=96, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Dimension of each attention head")
    parser.add_argument("--num_seqs", type=int, default=4, help="Number of sequences in the batch")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against PyTorch reference")
    return vars(parser.parse_args())


def ref_paged_attn(query, key_cache, value_cache, kv_lens, block_tables, scale):
    """Compute reference paged attention output using PyTorch."""
    num_seqs = query.shape[0]
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    for i in range(num_seqs):
        kv_len = kv_lens[i]
        q = query[i : i + 1] * scale  # (1, num_q_heads, head_dim)
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks].cpu().numpy()
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        gqa_ratio = q.shape[1] // k.shape[1]
        if gqa_ratio > 1:
            k = torch.repeat_interleave(k, gqa_ratio, dim=1)
            v = torch.repeat_interleave(v, gqa_ratio, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]

    torch.manual_seed(42)
    torch.set_default_device("cuda")

    kv_len = args["kv_len_per_rank"]
    num_heads = args["num_heads"]
    head_dim = args["head_dim"]
    num_seqs = args["num_seqs"]
    num_kv_heads = max(1, num_heads // 8)
    block_size = 1
    scale = head_dim**-0.5
    num_blocks_per_rank = (kv_len + block_size - 1) // block_size

    # Build input tensors
    query = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype)
    key_cache = torch.randn(num_blocks_per_rank, block_size, num_kv_heads, head_dim, dtype=dtype)
    value_cache = torch.randn(num_blocks_per_rank, block_size, num_kv_heads, head_dim, dtype=dtype)
    block_table = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(num_seqs, 1)
    kv_lens_tensor = torch.tensor([kv_len] * num_seqs, dtype=torch.int32)
    global_kv_lens = kv_lens_tensor.unsqueeze(0).repeat(world_size, 1)

    ctx.barrier()

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

    output = fd_layer(query, key_cache, value_cache, global_kv_lens, block_table)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(
            f"flash_decode: world_size={world_size}, num_heads={num_heads}, "
            f"head_dim={head_dim}, num_seqs={num_seqs}, kv_len_per_rank={kv_len}, dtype={dtype}"
        )

    if args["validate"]:
        # Gather all rank KV caches for a full reference computation
        all_key = torch.zeros(world_size * num_blocks_per_rank, block_size, num_kv_heads, head_dim, dtype=dtype)
        all_val = torch.zeros(world_size * num_blocks_per_rank, block_size, num_kv_heads, head_dim, dtype=dtype)
        dist.all_gather_into_tensor(all_key, key_cache)
        dist.all_gather_into_tensor(all_val, value_cache)

        ref_block_table = torch.cat([block_table + r * num_blocks_per_rank for r in range(world_size)], dim=-1)
        global_kv_len = kv_len * world_size
        ref_kv_lens = [global_kv_len] * num_seqs

        ref_output = ref_paged_attn(query, all_key, all_val, ref_kv_lens, ref_block_table, scale)

        try:
            torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
            if rank == 0:
                ctx.info(f"Validation passed: output[0,0,:3] = {output[0, 0, :3].tolist()}")
        except AssertionError as e:
            if rank == 0:
                ctx.info(f"Validation FAILED: {e}")
            raise

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
