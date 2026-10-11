# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Drive the PyTorch reference forward end-to-end: load real GPT-OSS-120B HF weights,
prefill a prompt, greedy-decode N tokens, print text. This validates the reference
math (the megakernel oracle) produces coherent output before we trust it.

Usage (on cluster, inside venv, GPU node):
  python run_reference.py --prompt "The capital of France is" --max-new 8
  python run_reference.py --layers 4 --selftest    # quick tiny-depth smoke
"""

from __future__ import annotations

import argparse
import time

import torch

from reference import GptOssConfig, build_yarn_rope, decode_forward
from load_hf import load_hf_weights


def make_kv_cache(cfg: GptOssConfig, n_layers: int, device, dtype=torch.float32):
    return [
        {
            "k": torch.zeros(cfg.max_seq_len, cfg.kv_dim, dtype=dtype, device=device),
            "v": torch.zeros(cfg.max_seq_len, cfg.kv_dim, dtype=dtype, device=device),
        }
        for _ in range(n_layers)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--layers", type=int, default=0, help="limit layers (0=all 36)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    device = args.device

    from tokenizer_util import load_tokenizer

    tok = load_tokenizer(args.snapshot)
    ids = tok.encode(args.prompt)
    print(f"prompt={args.prompt!r} ids={ids}")

    t0 = time.time()
    w = load_hf_weights(cfg, snapshot=args.snapshot, num_layers=L, device=device, dtype=torch.float32)
    # move weights to device
    for lw in w.layers:
        for k in lw:
            lw[k] = lw[k].to(device)
    w.embed = w.embed.to(device)
    w.final_norm = w.final_norm.to(device)
    w.lm_head = w.lm_head.to(device)
    print(f"loaded {L} layers in {time.time()-t0:.1f}s")

    cos, sin = build_yarn_rope(cfg, device=device)
    cfg_run = GptOssConfig()
    cfg_run.num_layers = L

    kv = make_kv_cache(cfg, L, device)

    # prefill
    pos = 0
    logits = None
    for tid in ids:
        h = w.embed[tid].float().to(device)
        logits = decode_forward(cfg_run, w, h, pos, kv, cos, sin)
        pos += 1

    out_ids = []
    for step in range(args.max_new):
        nxt = int(torch.argmax(logits))
        out_ids.append(nxt)
        h = w.embed[nxt].float().to(device)
        logits = decode_forward(cfg_run, w, h, pos, kv, cos, sin)
        pos += 1

    print("generated ids:", out_ids)
    print("generated text:", repr(tok.decode(out_ids)))


if __name__ == "__main__":
    main()
