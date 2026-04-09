#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: Fused Ring Prefill Attention

Demonstrates the flash_prefill_ring_layer module across 2 GPUs.
Uses mp.spawn (like examples/13_flash_decode/example_run.py).

Usage:
    python examples/14_flash_prefill_ring/example_run.py
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import iris
import iris.x
from flash_prefill_ring_layer import flash_prefill_ring_layer


def run_worker(rank, world_size, args):
    """Worker function for each GPU."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Initialize
    shmem = iris.iris()

    S_total = args["seq_len_total"]
    S_local = S_total // world_size
    H = args["num_q_heads"]
    H_kv = args["num_kv_heads"]
    D = args["head_dim"]
    scale = D ** -0.5

    device = f"cuda:{rank}"

    # Generate test data (rank 0 creates, all ranks broadcast)
    if rank == 0:
        q_full = torch.randn(S_total, H, D, dtype=torch.float16, device=device) / math.sqrt(D)
        k_full = torch.randn(S_total, H_kv, D, dtype=torch.float16, device=device) / math.sqrt(D)
        v_full = torch.randn(S_total, H_kv, D, dtype=torch.float16, device=device) / math.sqrt(D)
    else:
        q_full = torch.empty(S_total, H, D, dtype=torch.float16, device=device)
        k_full = torch.empty(S_total, H_kv, D, dtype=torch.float16, device=device)
        v_full = torch.empty(S_total, H_kv, D, dtype=torch.float16, device=device)

    q_full = torch.from_numpy(shmem.broadcast(q_full.cpu().numpy(), source_rank=0)).to(device)
    k_full = torch.from_numpy(shmem.broadcast(k_full.cpu().numpy(), source_rank=0)).to(device)
    v_full = torch.from_numpy(shmem.broadcast(v_full.cpu().numpy(), source_rank=0)).to(device)

    # Local slices
    q_local = q_full[rank * S_local:(rank + 1) * S_local].contiguous()

    # KV on sym heap
    k_sym = shmem.empty((S_local, H_kv, D), dtype=torch.float16)
    v_sym = shmem.empty((S_local, H_kv, D), dtype=torch.float16)
    k_sym.copy_(k_full[rank * S_local:(rank + 1) * S_local])
    v_sym.copy_(v_full[rank * S_local:(rank + 1) * S_local])

    shmem.barrier()

    # Create fused ring prefill layer
    layer = flash_prefill_ring_layer(
        shmem=shmem,
        rank=rank,
        num_ranks=world_size,
        num_q_heads=H,
        num_kv_heads=H_kv,
        head_dim=D,
        scale=scale,
        max_chunk_len=S_local,
    )

    # Run forward pass
    layer.clear_flags()
    out = layer(q_local, k_sym, v_sym, chunk_len=S_local)
    torch.cuda.synchronize()
    shmem.barrier()

    if rank == 0:
        print(f"[Example] Fused ring prefill: S_total={S_total}, H={H}, H_kv={H_kv}, D={D}")
        print(f"[Example] Output shape: {out.shape}, dtype: {out.dtype}")
        print(f"[Example] Output[0,0,:4]: {out[0, 0, :4].tolist()}")

    del shmem


def main():
    world_size = 2
    args = {
        "seq_len_total": 2048,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
    }

    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
