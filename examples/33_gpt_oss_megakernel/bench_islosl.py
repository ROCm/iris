# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Benchmark the GPT-OSS-120B megakernel across input/output length pairs.

For each (ISL, OSL):
  - TTFT  : time to process the ISL prompt tokens (prefill) and emit the first
            output token.
  - TPOT  : average per-token latency over the OSL decode steps (steady state).
  - E2E   : total wall-clock to produce all OSL tokens (TTFT + decode).

The megakernel decodes one token per launch, so prefill is ISL sequential steps.
The model is loaded once (quantized path) and reused across all configs.

Run:
  python bench_islosl.py --model /work/.../gptoss_120b.iris
  python bench_islosl.py            # load from HF cache
"""

from __future__ import annotations

import argparse
import time

import torch

from reference import GptOssConfig
from gpt_oss_120b_quantized_megakernel import MegaModel


def run_config(model, isl, osl, warmup=2):
    """Return (ttft_ms, tpot_ms, e2e_ms) for one (ISL, OSL) pair.

    A synthetic prompt of ISL tokens is fed through; the KV cache is reset by
    decoding from position 0. Token ids are arbitrary (timing is data-independent)."""
    cfg = model.cfg
    # reset KV cache and accumulators between configs
    model.kcache.zero_()
    model.vcache.zero_()

    prompt = [(i * 131 + 7) % cfg.vocab_size for i in range(isl)]

    # warm the kernels (compile + caches) on a throwaway short run
    for _ in range(warmup):
        model.step(prompt[0], 0)
    torch.cuda.synchronize()

    # ---- prefill: process the ISL prompt; time to first token = TTFT ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pos = 0
    nxt = None
    for tid in prompt:
        nxt = model.step(tid, pos)
        pos += 1
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1e3

    # ---- decode: OSL-1 further tokens, timed per step ----
    times = []
    for _ in range(osl - 1):
        t1 = time.perf_counter()
        nxt = model.step(nxt, pos)
        times.append((time.perf_counter() - t1) * 1e3)
        pos += 1
    tpot_ms = sum(times) / len(times) if times else ttft_ms
    e2e_ms = ttft_ms + sum(times)
    return ttft_ms, tpot_ms, e2e_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to a .iris weight file")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--configs", default="100:100,1024:100,1024:1024,2048:2048", help="comma list of ISL:OSL pairs")
    args = ap.parse_args()

    cfg = GptOssConfig()
    t0 = time.time()
    if args.model:
        model = MegaModel.from_iris(args.model, cfg, cfg.num_layers, quant=True)
    else:
        model = MegaModel(cfg, cfg.num_layers, snapshot=args.snapshot, quant=True)
    print(f"loaded model in {time.time() - t0:.1f}s (max_seq_len={cfg.max_seq_len})")

    rows = []
    for spec in args.configs.split(","):
        isl, osl = (int(x) for x in spec.split(":"))
        if isl + osl > cfg.max_seq_len:
            print(f"skip {isl}:{osl} (exceeds max_seq_len={cfg.max_seq_len})")
            continue
        ttft, tpot, e2e = run_config(model, isl, osl)
        thr = (isl + osl) / (e2e / 1e3)
        rows.append((isl, osl, ttft, tpot, e2e, thr))
        print(
            f"ISL={isl:5d} OSL={osl:5d}  TTFT={ttft:8.1f}ms  TPOT={tpot:6.2f}ms  "
            f"E2E={e2e:9.1f}ms  decode={1000.0 / tpot:6.1f} tok/s  throughput={thr:6.1f} tok/s"
        )

    print("\n| ISL | OSL | TTFT (ms) | TPOT (ms) | E2E (ms) | Decode (tok/s) |")
    print("| --- | --- | --------- | --------- | -------- | -------------- |")
    for isl, osl, ttft, tpot, e2e, thr in rows:
        print(f"| {isl} | {osl} | {ttft:.0f} | {tpot:.2f} | {e2e:.0f} | {1000.0 / tpot:.0f} |")


if __name__ == "__main__":
    main()
