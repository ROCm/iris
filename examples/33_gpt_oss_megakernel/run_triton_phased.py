# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
End-to-end GPT-OSS-120B decode using the Triton phase kernels, orchestrated by a
host layer-loop (one kernel launch per phase). This is the validated stepping
stone before fusing everything into the single persistent megakernel.

Run:
  python run_triton_phased.py --prompt "The capital of France is" --max-new 5
"""

from __future__ import annotations

import argparse
import time

import torch

import kernels as K
from reference import GptOssConfig, build_yarn_rope
from load_hf import load_hf_weights
from tokenizer_util import load_tokenizer


@torch.no_grad()
def decode_layer_triton(cfg, w, x, pos, kv, cos_row, sin_row, scale):
    H, DH, NH, NKV = cfg.hidden_dim, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    I = cfg.intermediate_dim
    for l in range(cfg.num_layers):
        lw = w.layers[l]
        xn = K.rmsnorm(x, lw["norm_attn"], cfg.rms_eps).to(torch.bfloat16)
        q = K.gemv(lw["w_q"], xn, lw["b_q"])
        k = K.gemv(lw["w_k"], xn, lw["b_k"])
        v = K.gemv(lw["w_v"], xn, lw["b_v"])
        K.rope_and_cache(q, k, v, cos_row, sin_row, kv[l]["k"], kv[l]["v"], pos, NH, NKV, DH)
        window = cfg.sliding_window if (l % 2 == 0) else 0
        attn = K.attention(q, kv[l]["k"], kv[l]["v"], lw["sinks"], pos, window, scale, NH, NKV, DH)
        o = K.gemv(lw["w_o"], attn.to(torch.bfloat16), lw["b_o"])
        x = x + o

        xn2 = K.rmsnorm(x, lw["norm_moe"], cfg.rms_eps).to(torch.bfloat16)
        logits = K.gemv(lw["router_w"], xn2, lw["router_b"])
        ids, gw = K.router_topk(logits, cfg.num_experts, cfg.top_k)
        moe = torch.zeros(H, dtype=torch.float32, device=x.device)
        for slot in range(cfg.top_k):
            e = int(ids[slot])
            guw = K.dequant_fp4(lw["gate_up_blocks"][e], lw["gate_up_scales"][e]).to(torch.bfloat16)  # [2I,H]
            gu = K.gemv(guw, xn2, lw["gate_up_b"][e])  # [2I]
            act = K.swiglu(gu, I, cfg.swiglu_alpha, cfg.swiglu_limit).to(torch.bfloat16)  # [I]
            dw = K.dequant_fp4(lw["down_blocks"][e], lw["down_scales"][e]).to(torch.bfloat16)  # [H,I]
            ev = K.gemv(dw, act, lw["down_b"][e])  # [H]
            moe += gw[slot] * ev
        x = x + moe
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    dev = "cuda"
    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    cfg.num_layers = L
    scale = 1.0 / (cfg.head_dim**0.5)

    tok = load_tokenizer(args.snapshot)
    ids = tok.encode(args.prompt)
    print(f"prompt={args.prompt!r} ids={ids}")

    t0 = time.time()
    w = load_hf_weights(GptOssConfig(), snapshot=args.snapshot, num_layers=L, device="cpu", dtype=torch.bfloat16)
    for lw in w.layers:
        for kk in lw:
            lw[kk] = lw[kk].to(dev)
    w.embed = w.embed.to(dev)
    w.final_norm = w.final_norm.to(dev)
    w.lm_head = w.lm_head.to(dev)
    print(f"loaded {L} layers in {time.time()-t0:.1f}s")

    cos, sin = build_yarn_rope(GptOssConfig(), device=dev)
    kv = [
        {
            "k": torch.zeros(cfg.max_seq_len, cfg.kv_dim, device=dev),
            "v": torch.zeros(cfg.max_seq_len, cfg.kv_dim, device=dev),
        }
        for _ in range(L)
    ]

    pos = 0
    last_logits = None
    for tid in ids:
        x = w.embed[tid].float()
        x = decode_layer_triton(cfg, w, x, pos, kv, cos[pos], sin[pos], scale)
        xf = K.rmsnorm(x, w.final_norm, cfg.rms_eps).to(torch.bfloat16)
        last_logits = K.gemv(w.lm_head, xf, None)
        pos += 1

    out = []
    for _ in range(args.max_new):
        nxt = K.argmax(last_logits)
        out.append(nxt)
        x = w.embed[nxt].float()
        x = decode_layer_triton(cfg, w, x, pos, kv, cos[pos], sin[pos], scale)
        xf = K.rmsnorm(x, w.final_norm, cfg.rms_eps).to(torch.bfloat16)
        last_logits = K.gemv(w.lm_head, xf, None)
        pos += 1

    print("generated ids:", out)
    print("generated text:", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
