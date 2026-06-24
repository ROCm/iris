#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""1 attention/tail GPU + 4 MoE GPUs GPT-OSS-120B batch-1 decode over iris.

Rank 0 runs attention + router + residual + LM head; ranks 1..4 each compute one of
the top-4 selected experts. Activations move attn<->moe via the iris symmetric heap
(iris.store / iris.load), synchronized with one all-rank shmem.barrier() per exchange
(correctness-first; device-flag pipelining is a later optimization).

Symmetric-heap invariant: a buffer allocated with the SAME sequence of shmem.zeros()
calls on every rank gets the SAME heap offset on every rank, so a remote store just
passes the local pointer + the destination rank. We therefore allocate the full
exchange buffer set (inbox + result + meta) on EVERY rank so the offsets line up,
even though each rank only uses its role's subset for compute.

Launch (5 ranks):
  python run_multi_gpu.py --model <iris> --prompt "The capital of France is" --max-new 5
  python run_multi_gpu.py --model <iris> --bench --tokens 32
Run from the example dir, PYTHONPATH including the iris repo root.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# iris.__init__ pulls in iris.ops, which imports the optional `tritonblas` package
# (collective GEMM kernels we do not use here). Stub it so the core load/store/heap
# API imports cleanly on environments without tritonblas installed.
import types as _types  # noqa: E402

if "tritonblas" not in sys.modules:
    _tb = _types.ModuleType("tritonblas")
    _tb_k = _types.ModuleType("tritonblas.kernels")
    _tb_s = _types.ModuleType("tritonblas.kernels.stages")
    for _name in ("GemmContext", "ScheduleContext", "make_tensor_view", "Tile"):
        setattr(_tb_s, _name, None)
    _tb.kernels = _tb_k
    _tb_k.stages = _tb_s
    sys.modules["tritonblas"] = _tb
    sys.modules["tritonblas.kernels"] = _tb_k
    sys.modules["tritonblas.kernels.stages"] = _tb_s

import iris  # noqa: E402

from reference import GptOssConfig, build_yarn_rope  # noqa: E402
from convert_to_iris import read_iris_header, load_iris_tensor  # noqa: E402
from tokenizer_util import load_tokenizer  # noqa: E402
from multi_gpu.protocol import ATTN_RANK, moe_rank  # noqa: E402
from multi_gpu import attn_kernels as ak  # noqa: E402
from multi_gpu import moe_kernels as mk  # noqa: E402

NWG = 180
LP = dict(BLOCK_K=1024, BLOCK_M=8, BLOCK_M_LM=16, BLOCK_NQ=32, BLOCK_ND=16,
          BLOCK_KQ=1024, MTILE=16, BLOCK_T=64, NSTAGES=3)


def _g(path, ents, nm, dev):
    return load_iris_tensor(path, ents[nm], device=dev)


def worker(local_rank, world_size, init_url, args):
    dist.init_process_group(
        backend="nccl", init_method=init_url, world_size=world_size, rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    dev = "cuda"
    cfg = GptOssConfig()
    H, qd, kvd = cfg.hidden_dim, cfg.q_dim, cfg.kv_dim
    NH, NKV, DH = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    E, TOPK, I, V = cfg.num_experts, cfg.top_k, cfg.intermediate_dim, cfg.vocab_size
    L = cfg.num_layers
    GU_NB, DN_NB = H // 32, I // 32
    NORMK = triton.next_power_of_2(H)
    eps = cfg.rms_eps
    scale = 1.0 / (DH ** 0.5)
    alpha, limit = cfg.swiglu_alpha, cfg.swiglu_limit
    max_seq = cfg.max_seq_len
    is_attn = rank == ATTN_RANK
    path, ents = args["model"], read_iris_header(args["model"])[1]

    def sym(*s, dt=torch.float32):
        return shmem.zeros(*s, device=dev, dtype=dt)

    # ---- exchange + scratch buffers FIRST, in an IDENTICAL order on every rank, so
    # the iris symmetric-heap offsets line up across ranks. A remote store then just
    # passes the local pointer + the destination rank. Allocating these before the
    # (role-specific, differently-sized) weights is essential: any rank-divergent
    # allocation BEFORE these would shift their heap offsets and break remote stores.
    nfp8 = sym(H, dt=torch.float8_e4m3fn)       # expert input (inbox on MoE ranks)
    nfp8_scl = sym(GU_NB, dt=torch.uint8)
    meta = sym(1, dt=torch.int32)                # selected expert id (inbox)
    gw1 = sym(1)                                 # gate weight (inbox)
    res = sym(TOPK * H)                          # gathered expert results (on attn rank)
    out = sym(H)                                 # this expert's result (on MoE ranks)
    x = sym(H); q = sym(qd); kk = sym(kvd); vv = sym(kvd)
    kcache = sym(L * max_seq * kvd); vcache = sym(L * max_seq * kvd)
    attn = sym(qd, dt=torch.bfloat16); o = sym(H)  # attn holds NH*DH = q_dim head outputs
    logits = sym(E); ids = sym(TOPK, dt=torch.int32); gw = sym(TOPK)
    gu = sym(2 * I); afp8 = sym(I, dt=torch.float8_e4m3fn); afp8_scl = sym(DN_NB, dt=torch.uint8)
    amax_v = sym(NWG); amax_i = sym(NWG, dt=torch.int32); next_tok = sym(1, dt=torch.int32)
    bar = sym(1, dt=torch.int32)

    # ---- weights (role-specific; loaded into regular torch memory, never remotely
    # addressed, so their allocation order does not affect the symmetric offsets) ----
    if is_attn:
        st = lambda key: torch.stack([_g(path, ents, f"L{l}.{key}", dev) for l in range(L)]).contiguous()
        W = {k: st(k) for k in ["norm_attn", "norm_moe", "w_q", "b_q", "w_k", "b_k", "w_v", "b_v",
                                "w_o", "b_o", "sinks", "router_w", "router_b"]}
        W["embed"] = _g(path, ents, "embed", dev)
        W["final_norm"] = _g(path, ents, "final_norm", dev)
        W["lm_head"] = _g(path, ents, "lm_head", dev)
        z1 = lambda n: torch.ones(L, n, dtype=torch.float32, device=dev)
        W["wq_s"], W["wk_s"], W["wv_s"], W["wo_s"], W["rw_s"] = z1(qd), z1(kvd), z1(kvd), z1(H), z1(E)
        cos, sin = build_yarn_rope(cfg, device=dev)
    else:
        st = lambda key: torch.stack([_g(path, ents, f"L{l}.{key}", dev) for l in range(L)]).contiguous()
        W = {k: st(k) for k in ["gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales"]}
        W["gu_b"] = st("gate_up_b").to(torch.bfloat16).contiguous()
        W["dn_b"] = st("down_b").to(torch.bfloat16).contiguous()

    heap = shmem.get_heap_bases()
    gridH = (triton.cdiv(H, 1024),)
    shmem.barrier()

    def decode_step(token_id, pos):
        if is_attn:
            x.copy_(W["embed"][token_id].float())
        shmem.barrier()
        for layer in range(L):
            if is_attn:
                bar.zero_()
                ak.attn_prologue_kernel[(NWG,)](
                    W["norm_attn"], W["norm_moe"],
                    W["w_q"], W["b_q"], W["w_k"], W["b_k"], W["w_v"], W["b_v"], W["w_o"], W["b_o"],
                    W["sinks"], W["router_w"], W["router_b"],
                    W["wq_s"], W["wk_s"], W["wv_s"], W["wo_s"], W["rw_s"],
                    x, q, kk, vv, kcache, vcache, attn, o, logits, ids, gw, nfp8, nfp8_scl,
                    cos[pos], sin[pos], bar, pos, scale, eps, layer, 0,
                    NWG=NWG, H=H, q_dim=qd, kv_dim=kvd, NH=NH, NKV=NKV, DH=DH,
                    E=E, TOPK=TOPK, SLIDING=cfg.sliding_window, GU_NB=GU_NB, max_seq=max_seq,
                    BLOCK_K=LP["BLOCK_K"], BLOCK_M=LP["BLOCK_M"], NORMK=NORMK,
                    BLOCK_T=LP["BLOCK_T"], NSTAGES=LP["NSTAGES"],
                    FP8_QKV=False, FP8_O=False, FP8_ROUTER=False, MXFP8_BLK=(1 << 30), num_warps=4,
                )
                for slot in range(TOPK):
                    ak.scatter_to_moe_kernel[gridH](
                        nfp8, nfp8_scl, ids, gw, nfp8, nfp8_scl, meta, gw1,
                        H=H, GU_NB=GU_NB, slot=slot, attn_rank=ATTN_RANK, dst_rank=moe_rank(slot),
                        BLOCK=1024, heap_bases=heap,
                    )
            shmem.barrier()

            if not is_attn:
                bar.zero_()
                mk.moe_expert_kernel[(NWG,)](
                    W["gate_up_blocks"], W["gate_up_scales"], W["gu_b"],
                    W["down_blocks"], W["down_scales"], W["dn_b"],
                    nfp8, nfp8_scl, meta, gw1, gu, afp8, afp8_scl, out, bar, 0,
                    layer, alpha, limit,
                    NWG=NWG, E=E, H=H, I=I, GU_NB=GU_NB, DN_NB=DN_NB,
                    BLOCK_NQ=LP["BLOCK_NQ"], BLOCK_ND=LP["BLOCK_ND"], BLOCK_KQ=LP["BLOCK_KQ"],
                    MTILE=LP["MTILE"], NSTAGES=LP["NSTAGES"], num_warps=4,
                )
                mk.scatter_back_kernel[gridH](
                    out, res, H=H, slot=rank - 1, moe_rank=rank, attn_rank=ATTN_RANK,
                    BLOCK=1024, heap_bases=heap,
                )
            shmem.barrier()

            if is_attn:
                ak.accumulate_kernel[gridH](res, x, o, H=H, TOPK=TOPK, BLOCK=1024)
            shmem.barrier()

        if is_attn:
            bar.zero_()
            ak.lm_head_kernel[(NWG,)](
                x, W["final_norm"], W["lm_head"], amax_v, amax_i, next_tok, bar, eps, 0,
                NWG=NWG, H=H, V=V, BLOCK_K=LP["BLOCK_K"], BLOCK_M_LM=LP["BLOCK_M_LM"],
                NORMK=NORMK, NSTAGES=LP["NSTAGES"], num_warps=4,
            )
        torch.cuda.synchronize()
        return int(next_tok.item()) if is_attn else None

    tok = load_tokenizer()
    enc = tok.encode(args["prompt"])
    ids_in = enc.ids if hasattr(enc, "ids") else enc

    if args["bench"]:
        # prefill (untimed)
        pos = 0
        nxt = None
        for t in ids_in:
            nxt = decode_step(t, pos); pos += 1
        shmem.barrier()
        # warmup
        for _ in range(4):
            nxt = decode_step(nxt if is_attn else 0, pos); pos += 1
        shmem.barrier()
        t0 = time.perf_counter()
        for _ in range(args["tokens"]):
            nxt = decode_step(nxt if is_attn else 0, pos); pos += 1
        shmem.barrier()
        dt = (time.perf_counter() - t0) / args["tokens"] * 1e3
        if is_attn:
            shmem.info(f"[multi-gpu 1+4] TPOT mean={dt:.2f} ms  ({1000.0/dt:.1f} tok/s)")
    else:
        pos = 0
        nxt = None
        for t in ids_in:
            nxt = decode_step(t, pos); pos += 1
        out_ids = [nxt]
        for _ in range(args["max_new"] - 1):
            # broadcast the attn rank's chosen token to all ranks so they step in lockstep
            tok_t = torch.tensor([nxt if is_attn else 0], device=dev, dtype=torch.int32)
            dist.broadcast(tok_t, src=ATTN_RANK)
            nxt = int(tok_t.item())
            nxt = decode_step(nxt, pos); pos += 1
            out_ids.append(nxt)
        if is_attn:
            shmem.info(f"generated ids: {out_ids}")
            shmem.info(f"generated text: {tok.decode([t for t in out_ids if t is not None])!r}")

    shmem.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--heap-size", type=int, default=1 << 34)
    ap.add_argument("--num-ranks", type=int, default=5)
    ap.add_argument("--port", type=int, default=0, help="rendezvous port (0 = pick a free one)")
    args = vars(ap.parse_args())
    port = args["port"]
    if port == 0:
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
    init_url = f"tcp://127.0.0.1:{port}"
    mp.spawn(worker, args=(args["num_ranks"], init_url, args), nprocs=args["num_ranks"], join=True)


if __name__ == "__main__":
    main()
