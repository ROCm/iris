#!/usr/bin/env python3
"""Which tiles are wrong, and in which pool did it happen?

The fused AR is wrong on the FIRST call with a fresh workspace, so this is a
race, not counter bookkeeping. Where the wrong tiles sit says which pool:

  wrong tiles ONLY in this rank's own M-shard  -> RS pool (it reduces exactly
                                                  that shard)
  wrong tiles spread across all shards         -> AG pool (it gathers every
                                                  shard from its owner)
  staged_c itself already wrong                -> GEMM pool / the .wt store

It checks all three levels on a single fresh call:
  1. staged_c vs the local partial       (did the GEMM pool write correctly?)
  2. scratch  vs the true reduced shard  (did the RS pool reduce correctly?)
  3. output   vs the full all-reduce     (did the AG pool gather correctly?)

Also runs ws=2 vs ws=8 -- a race that vanishes at ws=2 is a scaling
/contention effect rather than a plain indexing error.
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

N_GLOBAL = 2880
K_GLOBAL = 4096


def _worker(local_rank, world_size, init_url, M, bm, bn, tpf, split):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size,
        rank=local_rank, device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()

    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    dtype = torch.float16
    K_local = K_GLOBAL // world_size
    g, r_, a_ = split

    A = shmem.zeros((M, K_local), device="cuda", dtype=dtype)
    A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
    B = torch.randn(K_local, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}") * 0.1

    partial = torch.mm(A, B)              # what staged_c should hold on this rank
    full = partial.clone()
    dist.all_reduce(full, op=dist.ReduceOp.SUM)   # what output should hold
    torch.cuda.synchronize()

    m_per_rank = M // world_size
    my_rows = slice(rank * m_per_rank, (rank + 1) * m_per_rank)

    wsx = matmul_all_reduce_hbm_buffer_preamble(shmem, M, N_GLOBAL, dtype, bm, bn)
    shmem.barrier()
    out = torch.zeros(M, N_GLOBAL, dtype=dtype, device=f"cuda:{rank}")

    # exactly one call on a fresh workspace
    matmul_all_reduce_hbm_buffer(
        shmem, out, A, B, workspace=wsx, block_m=bm, block_n=bn, block_k=64,
        mfma=16, tiles_per_flag=tpf, num_gemm_sms=g, num_rs_sms=r_,
        num_ag_sms=a_, num_warps=8)
    torch.cuda.synchronize()
    shmem.barrier()

    d_staged = torch.abs(wsx["staged_c"] - partial).max().item()
    d_scratch = torch.abs(wsx["scratch"][my_rows] - full[my_rows]).max().item()
    d_out = torch.abs(out - full).max().item()

    # per-shard breakdown of the final output
    bad = (torch.abs(out - full) > 2.0)
    per_shard = [
        bad[s * m_per_rank:(s + 1) * m_per_rank].sum().item()
        for s in range(world_size)
    ]
    n_tiles_m = triton.cdiv(M, bm)

    for r in range(world_size):
        if r == rank and rank in (0, 1):
            print(f"\n[ws={world_size} rank={rank}] M={M} bm={bm} bn={bn} "
                  f"tpf={tpf} G/R/A={g}/{r_}/{a_}  m_tiles={n_tiles_m}")
            print(f"  1. staged_c vs local partial   maxdiff {d_staged:.4f}  "
                  f"{'GEMM POOL OK' if d_staged < 2.0 else 'GEMM POOL WRONG'}")
            print(f"  2. scratch[my shard] vs truth  maxdiff {d_scratch:.4f}  "
                  f"{'RS POOL OK' if d_scratch < 2.0 else 'RS POOL WRONG'}")
            print(f"  3. output vs full all-reduce   maxdiff {d_out:.4f}  "
                  f"{'AG POOL OK' if d_out < 2.0 else 'AG POOL WRONG'}")
            print(f"  bad elements per shard: {per_shard}")
            own = per_shard[rank]
            other = sum(per_shard) - own
            if sum(per_shard):
                print(f"  -> own shard {own}, other shards {other}  "
                      f"({'RS-side' if other == 0 else 'AG-side or both'})")
        shmem.barrier()

    dist.destroy_process_group()


def _free_port(explicit=None):
    if explicit:
        return explicit
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--num_ranks", type=int, default=8)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("-m", type=int, default=2048)
    p.add_argument("--bm", type=int, default=128)
    p.add_argument("--bn", type=int, default=128)
    p.add_argument("--tpf", type=int, default=1)
    p.add_argument("--split", type=str, default="96,96,64")
    a = p.parse_args()
    split = tuple(int(x) for x in a.split.split(","))
    mp.spawn(fn=_worker,
             args=(a.num_ranks, f"tcp://127.0.0.1:{_free_port(a.port)}",
                   a.m, a.bm, a.bn, a.tpf, split),
             nprocs=a.num_ranks, join=True)


if __name__ == "__main__":
    main()
