################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Part of the code adapted from
# https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_sp_decode_attn.py
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

import sys
from pathlib import Path
import pytest
from typing import List, Optional
from argparse import Namespace

import torch
import torch.distributed as dist
from examples.benchmark.reference.flash_decode_rccl.flash_decode_layer_rccl import flash_decode_layer_rccl

project_root = Path(__file__).resolve()
while not (project_root / "tests").is_dir() or not (project_root / "examples").is_dir():
    if project_root == project_root.parent:
        raise FileNotFoundError(
            "Could not find project root. Make sure your 'tests' and 'examples' "
            "directories are siblings in the project structure."
        )
    project_root = project_root.parent
print(f"Discovered Project Root: {project_root}")

module_dir = project_root / "examples" / "13_flash_decode"
print(f"Target Module Directory: {module_dir}")

target_file = module_dir / "fd_layer_rccl.py"
if module_dir.exists():
    sys.path.insert(0, str(module_dir))
    print(f"'{module_dir}' was added to sys.path.")
else:
    print("ERROR: Target directory not found. Not modifying sys.path.")

from utils import print_correctness_report  # noqa: E402


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens_per_rank: List[int],
    block_tables: torch.Tensor,
    scale: float,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_cpu = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len, kv_len = query_lens[i], kv_lens_per_rank[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            gqa_ratio = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, gqa_ratio, dim=1)
            v = torch.repeat_interleave(v, gqa_ratio, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=query.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if soft_cap is not None and soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)


def prepare_correctness_data(cfg, args, num_query_heads, num_kv_heads, num_blocks_total):
    """Creates data on Rank 0 and broadcasts it using torch.distributed."""
    head_dim = cfg["head_dim"]
    if args.rank == 0:
        query = torch.randn(cfg["num_seqs"], num_query_heads, head_dim, dtype=cfg["dtype"], device="cuda") / 10
        key_value_cache = (
            torch.randn(
                num_blocks_total, 2, cfg["block_size"], num_kv_heads, head_dim, dtype=cfg["dtype"], device="cuda"
            )
            / 10
        )
    else:
        query = torch.empty(cfg["num_seqs"], num_query_heads, head_dim, dtype=cfg["dtype"], device="cuda")
        key_value_cache = torch.empty(
            num_blocks_total, 2, cfg["block_size"], num_kv_heads, head_dim, dtype=cfg["dtype"], device="cuda"
        )

    dist.broadcast(query, src=0, group=args.tp_group)
    dist.broadcast(key_value_cache, src=0, group=args.tp_group)

    return {"query": query, "key_value_cache": key_value_cache}


@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_seqs", [1, 8])
@pytest.mark.parametrize("num_heads", [48, 96])
@pytest.mark.parametrize("kv_len", [4096, 65536])
def test_correctness_rccl_fused_full(kv_len, num_heads, num_seqs, head_dim):
    """
    Tests the correctness of the RCCL Fused implementation against the Torch reference.
    """
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    args = Namespace()
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.tp_group = dist.new_group(ranks=range(args.world_size))

    config = {
        "kv_len": kv_len,
        "num_heads": num_heads,
        "num_seqs": num_seqs,
        "head_dim": head_dim,
        "dtype": torch.float16,
        "block_size": 1,
        "soft_cap": 0.0,
    }

    # torch.manual_seed(42)

    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8 if num_query_heads >= 8 else 1
    scale = head_dim**-0.5
    num_blocks_per_rank = (config["kv_len"] + config["block_size"] - 1) // config["block_size"]
    num_blocks_total = num_blocks_per_rank * args.world_size

    tensor_data = prepare_correctness_data(config, args, num_query_heads, num_kv_heads, num_blocks_total)
    query = tensor_data["query"]
    key_value_cache = tensor_data["key_value_cache"]

    key_cache = key_value_cache[:, 0].contiguous()
    value_cache = key_value_cache[:, 1].contiguous()
    key_cache_this_rank = key_cache[args.rank * num_blocks_per_rank : (args.rank + 1) * num_blocks_per_rank]
    value_cache_this_rank = value_cache[args.rank * num_blocks_per_rank : (args.rank + 1) * num_blocks_per_rank]

    block_tables_this_rank = torch.arange(num_blocks_per_rank, dtype=torch.int32).repeat(num_seqs, 1).cuda()

    gathered_tables_list = [torch.empty_like(block_tables_this_rank) for _ in range(args.world_size)]
    dist.all_gather(gathered_tables_list, block_tables_this_rank, group=args.tp_group)
    ref_block_tables = torch.cat([tbl + r * num_blocks_per_rank for r, tbl in enumerate(gathered_tables_list)], dim=-1)

    keyword_params = {
        "page_size": config["block_size"],
        "scale": scale,
        "soft_cap": config["soft_cap"],
        "max_allowed_batch": config["num_seqs"],
    }
    fd_layer = flash_decode_layer_rccl(
        args.rank, args.world_size, num_query_heads, num_kv_heads, head_dim, head_dim, args.tp_group, **keyword_params
    )
    dist.barrier(group=args.tp_group)

    kv_lens_per_rank = [config["kv_len"]] * num_seqs
    kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32).cuda()
    global_kv_lens_tensor = kv_lens_tensor.unsqueeze(0).repeat(args.world_size, 1)

    output = fd_layer(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
    torch.cuda.synchronize()

    ref_output = ref_paged_attn(
        query=query.clone(),
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens_per_rank=[config["kv_len"] * args.world_size] * num_seqs,
        block_tables=ref_block_tables,
        scale=scale,
        soft_cap=config["soft_cap"],
    )
    dist.barrier(group=args.tp_group)

    error = None
    try:
        torch.testing.assert_close(output, ref_output, atol=1e-4, rtol=1e-4)
    except AssertionError as e:
        error = e

    print_correctness_report(args.rank, output, ref_output, error)

    if error:
        raise error
