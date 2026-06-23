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
python examples/33_gpt_oss_megakernel/bench_islosl.py --configs 100:100,1024:1024,2048:2048
```

`bench_tpot.py` reports steady-state time per output token (TPOT);
`bench_islosl.py` sweeps input/output length pairs and reports prefill and decode
latency separately. The quantized path runs at about 5.3 ms/token on MI355X; the
default BF16 path is slower but bit-faithful to the reference.

Measured on a single MI355X (quantized path, `max_seq_len = 4096`):

| ISL | OSL | TTFT (ms) | TPOT (ms) | End-to-end (ms) | Decode (tok/s) |
| --- | --- | --------- | --------- | --------------- | -------------- |
| 100 | 100 | 531 | 5.28 | 1053 | 189 |
| 1024 | 100 | 5397 | 5.28 | 5919 | 189 |
| 1024 | 1024 | 5399 | 5.28 | 10802 | 189 |
| 2048 | 2048 | 10803 | 5.28 | 21604 | 190 |

TPOT stays flat across context lengths because the decode attention is computed
with a blocked flash-decode. TTFT grows linearly with the input length: prefill
reuses the single-token decode kernel one prompt token at a time, so there is no
batched-prefill speedup (a batched prefill kernel would cut TTFT substantially).
The largest pair is bounded by `ISL + OSL <= max_seq_len`.

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
| `bench_islosl.py` | Prefill/decode benchmark across input/output length pairs. |
| `test_*.py` | Correctness tests. |

Developed on AMD MI355X (gfx950, ROCm 7.2, Triton 3.6).
