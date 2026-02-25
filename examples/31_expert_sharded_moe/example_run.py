#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Expert-sharded distributed MoE example using Iris.

Validates against a single-device reference implementation.

Usage:
    HIP_VISIBLE_DEVICES=0,1 python example_run.py
    HIP_VISIBLE_DEVICES=0,1 python example_run.py --num_ranks 2 --n_tokens 128 --d_model 64
"""

import os
import sys
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import iris

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expert_assignment import make_expt_dict_uniform, make_expt_assignment
from moe import mixture_of_expt_nosharded, mixture_of_expt_epsharded


def parse_args():
    parser = argparse.ArgumentParser(description="Expert-sharded MoE example with Iris")
    parser.add_argument("--num_ranks", type=int, default=2)
    parser.add_argument("--n_tokens", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_expts_tot", type=int, default=8)
    parser.add_argument("--n_expts_act", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


def run_worker(rank, world_size, init_url, args):
    dist.init_process_group(
        backend="nccl",
        init_method=init_url,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.cuda.set_device(rank)
    shmem = iris.iris()
    try:
        _run_moe_example(rank, world_size, shmem, args)
    finally:
        del shmem
        import gc

        gc.collect()
        dist.destroy_process_group()


def _run_moe_example(rank, world_size, shmem, args):
    n_tokens = args.n_tokens
    d_model = args.d_model
    n_expts_tot = args.n_expts_tot
    n_expts_act = args.n_expts_act
    n_tokens_local = n_tokens // world_size
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("=" * 60)
        print("Expert-Sharded MoE Example (Iris)")
        print("=" * 60)
        print(f"  ranks:        {world_size}")
        print(f"  n_tokens:     {n_tokens} ({n_tokens_local} per rank)")
        print(f"  d_model:      {d_model}")
        print(f"  n_expts_tot:  {n_expts_tot}")
        print(f"  n_expts_act:  {n_expts_act}")
        print(f"  expts/rank:   {n_expts_tot // world_size}")
        print()

    torch.manual_seed(0)
    x_global = torch.randn(n_tokens, d_model, device=device, dtype=torch.bfloat16)
    l_global = torch.rand(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    w_global = torch.randn(n_expts_tot, d_model, d_model, device=device, dtype=torch.bfloat16)
    b_global = torch.randn(n_expts_tot, d_model, device=device, dtype=torch.float32)

    dist.broadcast(x_global, src=0)
    dist.broadcast(l_global, src=0)
    dist.broadcast(w_global, src=0)
    dist.broadcast(b_global, src=0)

    n_shards = world_size
    expt_dict = make_expt_dict_uniform(n_shards, n_expts_tot)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device)

    if rank == 0:
        print("Expert assignment:")
        for s, expts in expt_dict.items():
            print(f"  rank {s}: experts {expts}")
        print()

    if rank == 0:
        print("Computing reference (non-sharded) MoE...")
    y_global_ref = mixture_of_expt_nosharded(
        x_global,
        l_global,
        w_global,
        b_global,
        n_expts_act,
    )

    first = rank * n_tokens_local
    last = first + n_tokens_local
    x_dp_local = x_global[first:last].contiguous()
    l_dp_local = l_global[first:last].contiguous()
    w_ep_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
    b_ep_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

    shmem.barrier()

    if rank == 0:
        print("Running expert-sharded MoE pipeline...")

    z_dp_local = mixture_of_expt_epsharded(
        x_dp_local,
        l_dp_local,
        w_ep_local,
        b_ep_local,
        expt_assignment,
        n_expts_act,
        shmem,
    )

    torch.cuda.synchronize()
    shmem.barrier()
    dist.barrier()

    y_global_tri = torch.empty_like(y_global_ref)
    dist.all_gather_into_tensor(y_global_tri, z_dp_local.contiguous())

    if rank == 0:
        print()
        print("--- Validation ---")
        print(f"  Reference output shape: {y_global_ref.shape}")
        print(f"  Sharded output shape:   {y_global_tri.shape}")
        print(f"  Reference first row[:5]: {y_global_ref[0, :5]}")
        print(f"  Sharded first row[:5]:   {y_global_tri[0, :5]}")
        print()

        diff = (y_global_ref.float() - y_global_tri.float()).abs()
        print(f"  max diff  = {diff.max().item():.6f}")
        print(f"  mean diff = {diff.mean().item():.6f}")
        print()

        try:
            torch.testing.assert_close(
                y_global_ref,
                y_global_tri,
                atol=args.atol,
                rtol=args.rtol,
            )
            print("PASSED: sharded MoE matches reference")
        except AssertionError as e:
            print(f"FAILED: {str(e)[:500]}")

        print("=" * 60)


def main():
    args = parse_args()
    assert args.n_tokens % args.num_ranks == 0, (
        f"n_tokens ({args.n_tokens}) must be divisible by num_ranks ({args.num_ranks})"
    )
    assert args.n_expts_tot % args.num_ranks == 0, (
        f"n_expts_tot ({args.n_expts_tot}) must be divisible by num_ranks ({args.num_ranks})"
    )

    init_url = "tcp://127.0.0.1:29504"
    mp.spawn(
        fn=run_worker,
        args=(args.num_ranks, init_url, args),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
