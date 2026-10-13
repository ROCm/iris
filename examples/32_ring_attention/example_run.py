#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Minimal example demonstrating ring attention using the RingAttention layer.

The sequence is split evenly across GPUs along the sequence dimension.
Each rank computes its share of the attention output.  After the ring passes
Q and V are combined via online-softmax, yielding the same result as a single
device running full attention on the entire sequence.

Usage::

    # Run on 2 GPUs (default)
    python examples/32_ring_attention/example_run.py

    # Run on 4 GPUs
    python examples/32_ring_attention/example_run.py --num_ranks 4

    # Non-causal (bidirectional) attention
    python examples/32_ring_attention/example_run.py --no_causal
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris
from ring_attention_layer import RingAttention


def parse_args():
    parser = argparse.ArgumentParser(description="Ring Attention example")
    parser.add_argument("--total_seq_len", type=int, default=4096, help="Total sequence length (split across GPUs)")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--num_ranks", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--no_causal", action="store_true", help="Use bidirectional (non-causal) attention")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Input tensor dtype",
    )
    return parser.parse_args()


def run(rank: int, world_size: int, init_url: str, args):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )

    shmem = iris.iris()
    torch.manual_seed(42)
    torch.set_default_device("cuda")

    dtype = getattr(torch, args.dtype)
    causal = not args.no_causal

    seq_local = args.total_seq_len // world_size
    num_heads = args.num_heads
    head_dim = args.head_dim

    if rank == 0:
        attn_type = "causal" if causal else "bidirectional"
        print(f"--- Ring Attention Example ({attn_type}) ---")
        print(f"  GPUs          : {world_size}")
        print(f"  Total seq len : {args.total_seq_len}")
        print(f"  Seq per GPU   : {seq_local}")
        print(f"  Heads × dim   : {num_heads} × {head_dim}")
        print(f"  dtype         : {dtype}")

    # Each rank creates its local Q, K, V chunk
    q = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
    k = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)
    v = torch.randn(seq_local, num_heads, head_dim, dtype=dtype)

    shmem.barrier()

    layer = RingAttention(shmem, num_heads=num_heads, head_dim=head_dim, causal=causal)

    # Warm-up pass
    _ = layer(q, k, v)
    torch.cuda.synchronize()
    shmem.barrier()

    # Timed pass
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output = layer(q, k, v)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    if rank == 0:
        print(f"\nOutput shape : {output.shape}")
        print(f"Output dtype : {output.dtype}")
        print(f"Elapsed time : {elapsed_ms:.2f} ms")
        print(f"Output[0, 0, :4] = {output[0, 0, :4].float()}")

    shmem.barrier()
    dist.destroy_process_group()


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    args = parse_args()
    init_url = f"tcp://127.0.0.1:{_find_free_port()}"
    mp.spawn(
        fn=run,
        args=(args.num_ranks, init_url, args),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
