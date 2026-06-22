<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# GPT-OSS 120B Quantized Megakernel (Iris / Triton)

A **single persistent Triton kernel** that runs **both attention and
mixture-of-experts for all 36 layers** of OpenAI's GPT-OSS-120B on **one GPU**,
for the batch-1 (decode) GEMV case.

This collapses [ROCm/cosmic](https://github.com/ROCm/cosmic)'s multi-GPU
hand-written assembly design — 1 GPU running an attention megakernel + 4 GPUs each
running a MoE megakernel (top-4 → 1 expert/GPU at batch-1) — into one Triton
megakernel on a single GPU. Attention and MoE are different *phases* of the same
resident kernel, synchronized with a grid-wide barrier; never separate launches,
never separate GPUs.

## Result

Loaded from the converted `.iris` weight file, real GPT-OSS-120B weights, greedy decode:

```
"The capital of France is"  ->  " Paris.\n\nGreat! If you have"   (token 12650 = " Paris")
"Q: What is 2+2? A:"        ->  " 4\n\nQ: What"
"The opposite of hot is"    ->  ' "cold."'
```

Byte-identical output to the PyTorch reference and the host-phased Triton driver.

## Architecture (per layer, all inside the persistent kernel)

```
RMSNorm → QKV+bias → NeoX YaRN RoPE → KV-cache append
       → GQA flash-decode (per-head attention SINK, alternating sliding-128/full window)
       → O-proj+bias + residual
       → RMSNorm → router top-4 (softmax-AFTER-topk)
       → 4× SwiGLU-OAI experts (MXFP4 weights, dequant in-kernel) → gate-weighted sum + residual
Then once: final RMSNorm → lm_head → argmax
```

Dims: 36 layers, hidden 2880, 64 Q / 8 KV heads (GQA), head_dim 64, 128 experts /
top-4, intermediate 2880, vocab 201088. SwiGLU-OAI α=1.702, limit=7. YaRN RoPE
θ=150000, factor=32. RMSNorm eps 1e-5.

## Precision

Non-expert weights (attention / router / embed / lm_head) are BF16. The 128 experts
are kept in **MXFP4** (E2M1 nibbles + per-32 E8M0 scales, the native HF format) — only
the 4 routed experts per token per layer are ever touched. Keeping experts in MXFP4 is
the memory-correct choice: a fully materialized BF16 model would be ~240 GB; the MXFP4
experts keep the whole model at ~65 GB.

Two compute paths for the expert GEMVs, selected by `--quant`:

- **BF16 (default).** FP4 weights are dequantized to BF16 in the GEMV inner loop and
  multiplied with BF16 activations (W4A16). Bit-faithful to the BF16 reference.
- **Quantized (`--quant`).** Activations are dynamically quantized to FP8-E4M3 (per-32
  E8M0, amax/448) and multiplied with the FP4 weights via `tl.dot_scaled`, which compiles
  to the native gfx950 **`v_mfma_scale_f32_16x16x128_f8f6f4`** tensor-core instruction
  (W4A8) — the same instruction cosmic hand-codes. Not bit-identical to BF16 (FP8
  activation quant), but the standard production deployment regime; output stays coherent.

## Performance (MI355X, 36 layers, TPOT = steady-state decode latency)

| Path | TPOT | tok/s |
| ---- | ---- | ----- |
| BF16 (dequant + scalar dot) | 83.6 ms | 12.0 |
| **Quantized (FP4×FP8 scaled MFMA)** | **29.4 ms** | **34.0** |

The quantized path is **2.8× faster** from using the native scaled-MFMA. This is a
correctness-first Triton implementation; cosmic's hand-tuned assembly reaches 1.70 ms/tok.
The remaining gap is kernel-level optimization (the serial program-0 phases — RMSNorm,
RoPE, top-k, residuals — plus weight prefetch / LDS pipelining), not the architecture.

## Files

| File | Purpose |
| ---- | ------- |
| `gpt_oss_120b_quantized_megakernel.py` | **The deliverable.** Single persistent `@triton.jit` megakernel + host driver (`MegaModel`). |
| `reference.py` | PyTorch fp32 reference forward — the numerical ground truth. |
| `load_hf.py` | Load HF safetensors into the reference layout; MXFP4 dequant helpers. |
| `convert_to_iris.py` | Convert HF checkpoint → a single `.iris` weight file (artifact). |
| `kernels.py` | Standalone Triton phase kernels (validated building blocks). |
| `tokenizer_util.py` | GPT-OSS tokenizer via the `tokenizers` lib (no `transformers` modeling dep). |
| `run_reference.py` | Drive the PyTorch reference end-to-end. |
| `run_triton_phased.py` | End-to-end with the phase kernels (host layer-loop) — validation stepping stone. |
| `test_kernels.py` | Per-kernel correctness vs reference (all cos≈1.0). |
| `test_barrier.py` | De-risk: grid-wide barrier inside one persistent kernel on ROCm. |
| `test_dot_scaled.py` | Validate `tl.dot_scaled` FP4×FP8 GEMV vs dequant reference. |
| `test_quant_expert.py` | Validate the quantized expert GEMV + FP8 activation quant. |
| `bench_tpot.py` | TPOT benchmark, BF16 vs quantized. |

## Run

```bash
# 1) Convert HF weights -> .iris artifact (once)
python convert_to_iris.py --out /path/gptoss_120b.iris

# 2) Generate with the megakernel
python gpt_oss_120b_quantized_megakernel.py \
    --model /path/gptoss_120b.iris \
    --prompt "The capital of France is" --max-new 8

# Or load straight from the HF cache (no .iris):
python gpt_oss_120b_quantized_megakernel.py --prompt "..." --max-new 8

# Quantized FP4xFP8 scaled-MFMA experts:
python gpt_oss_120b_quantized_megakernel.py --prompt "..." --max-new 8 --quant

# Benchmark TPOT (both paths):
python bench_tpot.py --tokens 32 --warmup 4

# Validate against the reference / phase kernels:
python test_kernels.py
python run_reference.py --prompt "..." --max-new 5
```

Developed and validated on **AMD MI355X (gfx950, 256 CUs, ROCm 7.2)**, Triton 3.6.

## How the single kernel stays synchronized

`grid = 256` (one persistent program per CU). Between phases, every program hits a
grid-wide barrier: a **monotonic** global counter that each program `atomic_add`s
(release), then spins on (acquire) until it reaches `(phase+1) * NUM_WG`. Targets
never reset, so there is no reset race. Per-phase work (GEMV rows, attention heads,
expert rows) is striped across the 256 programs. Because the grid size equals the
resident-workgroup capacity, every program is co-resident and the barrier cannot
deadlock.

## Quantized compute (`--quant`)

The expert GEMVs run on the native gfx950 scaled-MFMA via `tl.dot_scaled`:

- `_quant_act_fp8` dynamically quantizes the activation to FP8-E4M3 with per-32 E8M0
  scales (amax/448), matching cosmic.
- `_gemv_fp4_scaled` feeds FP4 weights (e2m1) × FP8 activations (e4m3) with their E8M0
  scales into `tl.dot_scaled`, which lowers to `v_mfma_scale_f32_16x16x128_f8f6f4`
  (verified in the compiled amdgcn — native, not bf16 emulation).

The output tiles M to the MFMA's 16-row minimum (only row 0 is the real token; the rest
are masked out). The reduction loops K in 128-wide blocks with masked tails (K=2880 is
not a multiple of 128).

Optimization headroom (toward cosmic's 1.70 ms/tok): parallelize the program-0 serial
phases, prefetch weights to LDS, and pipeline the MFMA K-loop.
