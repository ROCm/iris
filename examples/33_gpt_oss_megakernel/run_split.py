# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Split-kernel + CUDA-graph driver for the GPT-OSS-120B decode kernels.

Same weights, same buffers, same device functions as MegaModel; the grid barrier is
replaced by the kernel boundary. 7 phases per layer + 2 tail phases = 7*L+2 launches
per token (254 at L=36).

Those launches are captured as a CHAIN OF CHUNKED CUDA GRAPHS, not one graph: ROCm
graph REPLAY fails at 254 nodes with HSA_STATUS_ERROR_INVALID_PACKET_FORMAT while
capture succeeds, so build_graphs() emits ceil(L/chunk) graphs of 7*chunk nodes plus a
tail graph. Dispatch is paid once per graph rather than once per launch.

Correctness oracle: MegaModel itself. Same prompt, same positions -> identical token ids.
"""
from __future__ import annotations
import argparse, statistics, time
import torch
import triton

from gpt_oss_120b_quantized_megakernel import MegaModel, NUM_WG
import split_kernels as SK


class SplitModel(MegaModel):
    """Overrides only step(); allocation, packing and buffers are inherited unchanged."""

    def _launch(self, layer, phase, pos, dump_logits=False):
        cfg = self.cfg
        SK.gpt_oss_megakernel[(NUM_WG,)](
            self.norm_attn,
            self.norm_moe,
            self.wq,
            self.bq,
            self.wk,
            self.bk,
            self.wv,
            self.bv,
            self.wo,
            self.bo,
            self.sinks,
            self.router_w,
            self.router_b,
            self.wq_s,
            self.wk_s,
            self.wv_s,
            self.wo_s,
            self.router_w_s,
            self.gu_blk,
            self.gu_scl,
            self.gu_b,
            self.dn_blk,
            self.dn_scl,
            self.dn_b,
            self.final_norm,
            self.lm_head,
            self.x,
            self.normed,
            self.q,
            self.k,
            self.v,
            self.kcache,
            self.vcache,
            self.attn,
            self.o,
            self.logits,
            self.ids,
            self.gw,
            self.gu,
            self.act,
            self.moe,
            self.nfp8,
            self.nfp8_scl,
            self.afp8,
            self.afp8_scl,
            self.vlogits,
            self.amax_v,
            self.amax_i,
            self.next_tok,
            self.cos[pos],
            self.sin[pos],
            self.bar,
            pos,
            1.0 / (cfg.head_dim**0.5),
            cfg.rms_eps,
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
            L=self.L,
            H=cfg.hidden_dim,
            q_dim=cfg.q_dim,
            kv_dim=cfg.kv_dim,
            NH=cfg.num_heads,
            NKV=cfg.num_kv_heads,
            DH=cfg.head_dim,
            E=cfg.num_experts,
            TOPK=cfg.top_k,
            I=cfg.intermediate_dim,
            V=cfg.vocab_size,
            SLIDING=cfg.sliding_window,
            GU_NB=self.gu_nb,
            DN_NB=self.dn_nb,
            max_seq=cfg.max_seq_len,
            BLOCK_K=1024,
            BLOCK_KI=256,
            BLOCK_M=8,
            BLOCK_M_LM=16,
            NORMK=triton.next_power_of_2(cfg.hidden_dim),
            QUANT=self.quant,
            BLOCK_NQ=32,
            BLOCK_ND=16,
            BLOCK_KQ=1024,
            MTILE=16,
            BLOCK_T=64,
            NSTAGES=3,
            FP8_QKV=("qkv" in self.fp8_components),
            FP8_O=("o" in self.fp8_components),
            FP8_ROUTER=("router" in self.fp8_components),
            MXFP8_BLK=self.fp8_scale_blk,
            DUMP_LOGITS=dump_logits,
            layer=layer,
            PHASE=phase,
            num_warps=4,
        )

    def step(self, token_id: int, pos: int, dump_logits: bool = False) -> int:
        # DELIBERATE: MegaModel.step() calls _check_grid_fits(), which refuses
        # NUM_WG > the device CU count. That guard exists because a GRID-WIDE BARRIER needs
        # every program resident to complete -- above the CU count the unscheduled programs
        # never arrive and the resident ones spin forever (measured: 256 runs, 257 hangs on
        # MI355X). These kernels have NO barrier, so residency does not bound the grid and a
        # wider-than-CU grid is merely oversubscribed, not deadlocked.
        #
        # Not inheriting it is not just safe, it is necessary: the grid width is a LEVER here.
        # A barrier-free ablation of the fused kernel ran 2.11 ms at NWG=768 against 2.47 ms
        # at NWG=180, and a pure streaming read reaches only ~44% of device bandwidth at 180.
        # The guard encodes a limit that exists only in the design it came from.
        self.x.copy_(self.embed[token_id].float())
        for layer in range(self.L):
            for ph in range(7):
                self._launch(layer, ph, pos, dump_logits)
        self._launch(0, 7, pos, dump_logits)
        self._launch(0, 8, pos, dump_logits)
        return int(self.next_tok.item())

    # ---- graph-captured. ROCm REPLAY (not capture) dies at 254 nodes with a malformed
    # AQL packet, so the token is captured as a CHAIN OF CHUNK GRAPHS of `chunk` layers
    # each (7*chunk nodes) plus a tail graph. Replays/token = ceil(L/chunk) + 1, versus
    # 7L+2 eager launches. ----
    def build_graphs(self, pos: int, chunk: int):
        def cap(fn):
            fn()                       # warm outside capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            return g
        graphs = []
        for lo in range(0, self.L, chunk):
            hi = min(lo + chunk, self.L)
            graphs.append(cap(lambda lo=lo, hi=hi: [self._launch(l, p, pos)
                                                    for l in range(lo, hi) for p in range(7)]))
        graphs.append(cap(lambda: [self._launch(0, 7, pos), self._launch(0, 8, pos)]))
        return graphs


def _gen(model, ids, n_new):
    pos, nxt = 0, None
    for tid in ids:
        nxt = model.step(tid, pos); pos += 1
    out = [nxt]
    for _ in range(n_new - 1):
        nxt = model.step(nxt, pos); pos += 1; out.append(nxt)
    return out


def main():
    from reference import GptOssConfig
    from tokenizer_util import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--quant", action="store_true", default=True)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--baseline-ms", type=float, default=0.0)
    args = ap.parse_args()

    chunk = args.chunk
    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    tok = load_tokenizer(args.snapshot)
    ids = tok.encode(args.prompt)
    print("RUN-HEADER prompt=%r ids=%s layers=%d quant=%s" % (args.prompt, ids, L, args.quant), flush=True)

    # ONE PROCESS, BOTH FACTS. Two live MegaModel instances interleaving launches is
    # what crashes the runtime (it dies on a plain fused.step() AFTER the split model has
    # run), so the fused model is finished and FREED before the split model is built.
    t0 = time.time()
    fused = MegaModel(cfg, L, snapshot=args.snapshot, quant=args.quant)
    print("loaded fused in %.1fs" % (time.time() - t0), flush=True)
    ref = _gen(fused, ids, args.max_new)
    print("FUSED  ids: %s  %r" % (ref, tok.decode(ref)), flush=True)
    pos = len(ids)
    ftok_ref = fused.step(ref[-1], pos)
    print("  [1/3] timing fused ...", flush=True)

    def bench(fn, reps, warm=5):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
        s_.record()
        for _ in range(reps): fn()
        e_.record(); torch.cuda.synchronize()
        return s_.elapsed_time(e_) / reps

    t_fused = statistics.median([bench(lambda: fused.step(ref[-1], pos), args.reps) for _ in range(3)])
    print("        fused %.4f ms   (reference token %d)" % (t_fused, ftok_ref), flush=True)
    del fused
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print("        fused model FREED", flush=True)

    split = SplitModel(cfg, L, snapshot=args.snapshot, quant=args.quant)
    got = _gen(split, ids, args.max_new)
    print("SPLIT  ids: %s  %r" % (got, tok.decode(got)), flush=True)
    print("  GATE token-identity (eager): %s" % ("MATCH" if got == ref else "MISMATCH"), flush=True)
    if got != ref:
        return
    # Eager timing is deliberately NOT run here: ~13,000 launches immediately before a
    # capture reliably precedes a runtime crash, and the eager figure is not under test.
    chunk = args.chunk
    print("  [3/3] chunked graphs, chunk=%d (%d nodes/graph) ..." % (chunk, 7 * chunk), flush=True)
    gs = split.build_graphs(pos, chunk)
    def run_all():
        for g in gs: g.replay()
    t_graph = statistics.median([bench(run_all, args.reps) for _ in range(3)])
    print("        %d replays/token : %.4f ms" % (len(gs), t_graph), flush=True)
    split.next_tok.zero_(); split.x.copy_(split.embed[ref[-1]].float())
    run_all(); torch.cuda.synchronize()
    gtok = int(split.next_tok.item())

    print("\n  GATE graph-replay token: graph=%d fused=%d  %s"
          % (gtok, ftok_ref, "OK" if gtok == ftok_ref else "MISMATCH -> TIMING IS VOID"), flush=True)
    if gtok != ftok_ref:
        return
    print("\n  %-28s %8s   %s" % ("config", "ms/token", "vs fused"))
    print("  %-28s %8.4f   %s" % ("fused megakernel", t_fused, "1.00x"))
    print("  %-28s %8.4f   %.2fx   (chunk=%d)" % ("split + chunked graphs", t_graph, t_fused / t_graph, chunk))
    print("\n  launches/token: 7*%d + 2 = %d" % (L, 7 * L + 2))
    # Baseline comparison is opt-in via --baseline-ms. A competitive figure about AMD
    # silicon should not be a literal compiled into the source; that is a disclosure call,
    # not a code default.
    if args.baseline_ms > 0:
        print("  baseline %.4f ms  ->  split+graph is %.2fx" % (args.baseline_ms, args.baseline_ms / t_graph), flush=True)


if __name__ == "__main__":
    main()
