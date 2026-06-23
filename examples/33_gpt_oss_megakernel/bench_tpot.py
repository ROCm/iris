# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Benchmark TPOT (time per output token) for the GPT-OSS-120B megakernel.

TPOT = steady-state decode latency: prefill the prompt (not timed), then time
each subsequent single-token decode step and average (after warmup). Reports
BF16-dequant vs quantized (FP4xFP8 native scaled-MFMA) paths.

Run:
  python bench_tpot.py --model /work/.../gptoss_120b.iris --tokens 32 --warmup 4
  python bench_tpot.py --tokens 32              # load from HF cache
"""

from __future__ import annotations

import argparse
import time

import torch

from reference import GptOssConfig
from gpt_oss_120b_quantized_megakernel import MegaModel
from tokenizer_util import load_tokenizer


def bench(model, ids, n_tokens, warmup):
    pos = 0
    nxt = None
    for tid in ids:  # prefill (untimed)
        nxt = model.step(tid, pos)
        pos += 1
    # warmup decode steps
    for _ in range(warmup):
        nxt = model.step(nxt, pos)
        pos += 1
    # timed decode steps
    torch.cuda.synchronize()
    times = []
    for _ in range(n_tokens):
        t0 = time.perf_counter()
        nxt = model.step(nxt, pos)  # step() already synchronizes
        times.append((time.perf_counter() - t0) * 1e3)  # ms
        pos += 1
    times.sort()
    mean = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
    return mean, p50, p99, times[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--modes", default="bf16,quant", help="comma list: bf16,quant")
    ap.add_argument("--fp8-attn", action="store_true", help="store attention/router weights in FP8")
    args = ap.parse_args()

    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    tok = load_tokenizer(args.snapshot)
    ids = tok.encode(args.prompt)

    for mode in args.modes.split(","):
        quant = mode == "quant"
        t0 = time.time()
        if args.model:
            model = MegaModel.from_iris(args.model, cfg, L, quant=quant, fp8_attn=args.fp8_attn)
        else:
            model = MegaModel(cfg, L, snapshot=args.snapshot, quant=quant, fp8_attn=args.fp8_attn)
        load_s = time.time() - t0
        mean, p50, p99, best = bench(model, ids, args.tokens, args.warmup)
        print(
            f"[{mode:5s}] L={L} load={load_s:.1f}s  TPOT mean={mean:.2f}ms "
            f"p50={p50:.2f}ms p99={p99:.2f}ms best={best:.2f}ms  "
            f"({1000.0/mean:.1f} tok/s)"
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
