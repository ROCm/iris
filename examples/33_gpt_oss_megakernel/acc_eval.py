"""Accuracy comparison: FP8-attention vs BF16-attention megakernel.

Both share the FP4 experts; this isolates the error introduced by quantizing the
attention/router weights to FP8. We teacher-force BOTH models on the SAME token
streams (greedy from the BF16 model) so the per-position comparison is apples to
apples (greedy divergence would otherwise cascade after one mismatch and overstate
error). At each decode position we dump full logits from both and compute:

  top1   : fraction of positions where argmax agrees
  top5   : mean overlap of the top-5 token sets
  top10  : mean overlap of the top-10 token sets
  KL     : KL(P_bf16 || P_fp8) over softmax(logits) [nats]
  cos    : cosine similarity of the raw logit vectors
  drank  : mean rank of the BF16 top-1 token within the FP8 ordering (0 = same)

Run:
  python acc_eval.py --model <iris> --tokens 48
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from reference import GptOssConfig
from gpt_oss_120b_quantized_megakernel import MegaModel

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "def fibonacci(n):",
    "The mitochondria is the",
    "In 1492, Columbus",
    "The chemical symbol for gold is",
    "import numpy as np\n",
    "She walked into the room and",
    "The square root of 144 is",
    "Water boils at a temperature of",
    "Dear Sir or Madam,\n",
    "The three branches of the US government are",
    "To make a peanut butter sandwich, first",
    "The theory of relativity was developed by",
    "for i in range(10):\n    print(",
    "The opposite of hot is",
]


def topk_overlap(a, b, k):
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    return len(sa & sb) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=48, help="decode steps per prompt")
    ap.add_argument("--mode", default="fp8attn", choices=["fp8attn","fp4experts"],
                    help="fp8attn: FP8 vs BF16 attn (FP4 experts both); fp4experts: FP4 vs BF16 experts (BF16 attn both)")
    ap.add_argument("--components", default=None,
                    help="comma list of FP8 components for the TEST model in fp8attn mode: qkv,o,router")
    ap.add_argument("--fp8-scale-blk", type=int, default=0,
                    help="FP8 weight scale block size along K (0=per-row, 32=MXFP8)")
    args = ap.parse_args()

    cfg = GptOssConfig()
    from tokenizer_util import load_tokenizer

    tok = load_tokenizer()

    if args.mode == "fp8attn":
        comps = set(args.components.split(",")) if args.components else {"qkv", "o", "router"}
        blk = args.fp8_scale_blk
        print(f"ref = BF16-attn + FP4-experts; test = FP8 {sorted(comps)} scale_blk={blk or 'per-row'} + FP4-experts")
        ref = MegaModel.from_iris(args.model, cfg, cfg.num_layers, quant=True, fp8_attn=False)
        test = MegaModel.from_iris(args.model, cfg, cfg.num_layers, quant=True, fp8_components=comps, fp8_scale_blk=blk)
    else:
        print("ref = BF16-attn + BF16-experts; test = BF16-attn + FP4-experts")
        ref = MegaModel.from_iris(args.model, cfg, cfg.num_layers, quant=False, fp8_attn=False)
        test = MegaModel.from_iris(args.model, cfg, cfg.num_layers, quant=True, fp8_attn=False)

    # accumulators
    n = 0
    top1 = 0
    top5 = 0.0
    top10 = 0.0
    kl_sum = 0.0
    cos_sum = 0.0
    drank_sum = 0.0
    greedy_match = 0  # would free-running greedy pick the same token?

    for pi, prompt in enumerate(PROMPTS):
        enc = tok.encode(prompt)
        ids = enc.ids if hasattr(enc, "ids") else enc

        # reset KV caches for both
        ref.kcache.zero_(); ref.vcache.zero_()
        test.kcache.zero_(); test.vcache.zero_()

        # prefill both on the prompt
        pos = 0
        rt = None
        for t in ids:
            rt = ref.step(t, pos)
            test.step(t, pos)
            pos += 1

        # teacher-forced decode: feed BOTH the BF16 greedy token each step
        cur = rt
        for _ in range(args.tokens):
            r_tok, r_log = ref.step(cur, pos, dump_logits=True)
            t_tok, t_log = test.step(cur, pos, dump_logits=True)
            r_log = r_log.float()
            t_log = t_log.float()

            n += 1
            if r_tok == t_tok:
                top1 += 1
            top5 += topk_overlap(r_log, t_log, 5)
            top10 += topk_overlap(r_log, t_log, 10)
            p = F.log_softmax(r_log, dim=0)
            q = F.log_softmax(t_log, dim=0)
            kl_sum += F.kl_div(q, p, log_target=True, reduction="sum").item()
            cos_sum += F.cosine_similarity(r_log, t_log, dim=0).item()
            # rank of bf16 top-1 in fp8 ordering
            order = torch.argsort(t_log, descending=True)
            drank_sum += (order == r_tok).nonzero()[0, 0].item()
            # also track if free-running greedy would agree (informational)
            if r_tok == t_tok:
                greedy_match += 1

            cur = r_tok  # teacher force with reference token
            pos += 1
        print(f"  [{pi+1}/{len(PROMPTS)}] {prompt[:40]!r:42}  running top1={top1/n:.4f}")

    print(f"\n==== accuracy comparison [mode={args.mode}] (teacher-forced) ====")
    print(f"positions evaluated : {n}")
    print(f"top-1 agreement     : {top1/n:.4f}")
    print(f"top-5 overlap       : {top5/n:.4f}")
    print(f"top-10 overlap      : {top10/n:.4f}")
    print(f"KL(bf16||fp8) nats  : {kl_sum/n:.5f}")
    print(f"logit cosine        : {cos_sum/n:.5f}")
    print(f"mean rank of bf16 #1: {drank_sum/n:.3f}  (0 = identical argmax)")


if __name__ == "__main__":
    main()
