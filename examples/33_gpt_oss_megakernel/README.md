<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# GPT-OSS-120B Megakernel

A single persistent Triton kernel that runs the full GPT-OSS-120B decode step —
both attention and the mixture-of-experts, for every layer — on one GPU. The
kernel is launched once per token and loops over the layers internally, so the
whole model forward pass is one resident kernel rather than a sequence of
launches.

The implementation lives in
`gpt_oss_120b_quantized_megakernel.py`. A PyTorch reference
(`reference.py`) defines the expected numerics, and the standalone Triton
building blocks are in `kernels.py`.

## Model

GPT-OSS-120B is a sparse mixture-of-experts model: 36 layers, hidden size 2880,
64 query / 8 key-value heads, 128 experts with top-4 routing, SwiGLU experts,
grouped-query attention with attention sinks, and alternating sliding / full
attention windows. Attention, router, embedding and LM-head weights are BF16; the
experts are stored in MXFP4 (4-bit) and only the routed experts are read per step,
which keeps the resident model near 65 GB.

## Usage

Convert a HuggingFace checkpoint to a single weight file, then generate:

```terminal
python examples/33_gpt_oss_megakernel/convert_to_iris.py --out gptoss_120b.iris
python examples/33_gpt_oss_megakernel/gpt_oss_120b_quantized_megakernel.py \
    --model gptoss_120b.iris --prompt "The capital of France is" --max-new 8
```

The weights can also be read straight from the HuggingFace cache by omitting
`--model`. Pass `--quant` to run the experts with the FP4 x FP8 scaled
matrix-multiply path instead of dequantizing to BF16.

## Validation

The Triton kernels are checked against the PyTorch reference:

```terminal
python examples/33_gpt_oss_megakernel/test_kernels.py
python examples/33_gpt_oss_megakernel/run_reference.py --prompt "The capital of France is"
```

## Benchmarking

```terminal
python examples/33_gpt_oss_megakernel/bench_tpot.py --tokens 32 --warmup 4
```

Reported as time per output token (TPOT). On MI355X the quantized path runs at
roughly 9.4 ms/token; the default BF16 path is slower but bit-faithful to the
reference.

## Files

| File | Purpose |
| ---- | ------- |
| `gpt_oss_120b_quantized_megakernel.py` | The persistent megakernel and its host driver. |
| `reference.py` | PyTorch reference forward pass. |
| `kernels.py` | Standalone Triton building-block kernels. |
| `load_hf.py` | Read HuggingFace weights into the reference layout. |
| `convert_to_iris.py` | Convert a checkpoint to a single weight file. |
| `tokenizer_util.py` | Tokenizer wrapper. |
| `run_reference.py`, `run_triton_phased.py` | End-to-end drivers. |
| `bench_tpot.py` | Decode-latency benchmark. |
| `test_*.py` | Correctness tests. |

Developed on AMD MI355X (gfx950, ROCm 7.2, Triton 3.6).
