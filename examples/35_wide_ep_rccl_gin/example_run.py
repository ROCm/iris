#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Wide-EP MoE layer across nodes: Iris intra-node, RCCL GIN inter-node.

Each rank holds ``num_experts / world`` experts and a batch of tokens routed
top-k over *all* experts, so most tokens leave the node. One MoE step is

    dispatch -> local experts -> combine

and the result is checked against a reference every rank computes alone:
expert weights are generated from a seed, so each rank can apply any expert
to its own tokens without communicating.

Launch one torchrun per node (see README.md), e.g. for 2 nodes x 8 GPUs:

    torchrun --nnodes 2 --node_rank $NODE --nproc_per_node 8 \\
        --master_addr $HEAD --master_port 29500 example_run.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from iris.experimental.rccl_gin import RcclGinProvider
from iris.experimental.wide_ep import WideEPBuffer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=512, help="tokens per rank")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--experts_per_rank", type=int, default=4)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--iters", type=int, default=3, help="checked MoE steps")
    p.add_argument("--bench_iters", type=int, default=20, help="timed dispatch+combine steps (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def expert_weight(e: int, hidden: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed * 100_003 + e)
    return (torch.randn(hidden, hidden, generator=g, device=device) / hidden**0.5).to(torch.bfloat16)


def route(tokens: int, num_experts: int, topk: int, gen, device):
    """Uniform top-k routing over every expert, softmax weights."""
    scores = torch.rand(tokens, num_experts, generator=gen, device=device)
    weights, idx = scores.topk(topk, dim=1)
    return idx, torch.softmax(weights, dim=1)


def run_local_experts(recv_x, recv_expert, recv_weight, weights):
    """y[i] = sum_j recv_weight[i, j] * (recv_x[i] @ W[recv_expert[i, j]])."""
    y = torch.zeros(recv_x.shape, dtype=torch.float32, device=recv_x.device)
    for e, w in enumerate(weights):
        row, slot = (recv_expert == e).nonzero(as_tuple=True)
        if row.numel():
            y.index_add_(0, row, (recv_x[row] @ w).float() * recv_weight[row, slot, None])
    return y.to(recv_x.dtype)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    num_experts = args.experts_per_rank * world

    provider = RcclGinProvider()
    buf = WideEPBuffer(provider, num_experts, args.hidden, args.tokens, args.topk)
    nodes = world // provider.lsa_size
    if rank == 0:
        print(
            f"wide EP: {world} ranks on {nodes} node(s) (LSA team {provider.lsa_size}), "
            f"{num_experts} experts, top-{args.topk}, {args.tokens} tokens/rank, hidden {args.hidden}; "
            f"RCCL {provider.rccl_version}, GIN {provider.gin_type_name}, "
            f"window {buf.window_bytes / 2**20:.0f} MiB",
            flush=True,
        )

    all_w = [expert_weight(e, args.hidden, args.seed, dev) for e in range(num_experts)]
    my_w = all_w[rank * args.experts_per_rank : (rank + 1) * args.experts_per_rank]
    gen = torch.Generator(device=dev).manual_seed(args.seed * 7919 + rank)

    direct = torch.tensor(buf.window.address_map.direct, device=dev)
    ok = True
    remote_frac = 0.0
    for it in range(args.iters):
        x = torch.randn(args.tokens, args.hidden, generator=gen, device=dev).to(torch.bfloat16)
        topk_idx, topk_w = route(args.tokens, num_experts, args.topk, gen, dev)

        recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, topk_idx, topk_w)
        y = run_local_experts(recv_x, recv_expert, recv_weight, my_w)
        out = buf.combine(y, handle)

        ref = torch.zeros(args.tokens, args.hidden, dtype=torch.float32, device=dev)
        for j in range(args.topk):
            for e in topk_idx[:, j].unique().tolist():
                rows = (topk_idx[:, j] == e).nonzero().flatten()
                ref[rows] += (x[rows] @ all_w[e]).float() * topk_w[rows, j, None]
        err = (out.float() - ref).abs().max().item()
        scale = ref.abs().max().item()
        good = err <= 2e-2 * scale + 1e-2
        ok &= good
        # A token copy crossed nodes iff its destination is not direct in the map.
        remote_frac = (~direct)[handle.dst].float().mean().item()
        if not good or rank == 0:
            print(
                f"[rank {rank}] step {it}: received {sum(handle.recv_counts)} rows, "
                f"max err {err:.3e} (ref max {scale:.3e}) {'ok' if good else 'MISMATCH'}",
                flush=True,
            )

    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"WIDE_EP_RESULT: {'PASS' if flag.item() else 'FAIL'} "
            f"({args.iters} steps, {remote_frac:.0%} of rank 0's token copies crossed nodes)",
            flush=True,
        )

    if args.bench_iters and flag.item():
        x = torch.randn(args.tokens, args.hidden, generator=gen, device=dev).to(torch.bfloat16)
        topk_idx, topk_w = route(args.tokens, num_experts, args.topk, gen, dev)
        times = {"dispatch": [], "combine": []}
        for i in range(args.bench_iters + 3):
            provider.barrier()
            t0, t1, t2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            t0.record()
            recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, topk_idx, topk_w)
            t1.record()
            out = buf.combine(recv_x, handle)
            t2.record()
            torch.cuda.synchronize()
            if i >= 3:
                times["dispatch"].append(t0.elapsed_time(t1))
                times["combine"].append(t1.elapsed_time(t2))
        sent = handle.token.numel() * args.hidden * 2  # bytes of token data this rank sent
        median = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        stats = torch.tensor([median["dispatch"], median["combine"], sent], dtype=torch.float64)
        gathered = [torch.zeros_like(stats) for _ in range(world)]
        dist.all_gather(gathered, stats)
        if rank == 0:
            g = torch.stack(gathered)
            d_ms, c_ms = g[:, 0].max().item(), g[:, 1].max().item()
            gb = g[:, 2].mean().item() / 1e9
            print(
                f"bench (median of {args.bench_iters}, slowest rank): dispatch {d_ms:.3f} ms, combine {c_ms:.3f} ms; "
                f"{gb * 1e3:.1f} MB token data per rank each way -> "
                f"{gb / (d_ms / 1e3):.2f} / {gb / (c_ms / 1e3):.2f} GB/s per rank (incl. host-side layout)",
                flush=True,
            )

    buf.close()
    provider.close()
    dist.destroy_process_group()
    sys.exit(0 if flag.item() else 1)


if __name__ == "__main__":
    main()
