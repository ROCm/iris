# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Reusable GPT-OSS device ops (Triton @triton.jit helpers), one module per
(op, dtype). The single-GPU megakernel and the planned per-GPU multi-GPU kernels
(1 attention/tail GPU + 4 MoE GPUs) import the same ops from here.

Files:
  barrier.py    grid-wide relaxed-poll barrier
  fp4.py        E2M1 magnitude LUT
  quant.py      fused residual+RMSNorm+FP8 activation quant
  rmsnorm.py    residual+RMSNorm materialize (bf16)
  gemv_bf16.py  bf16-weight GEMV (tiled, +rmsnorm, +resid_rmsnorm, scalar)
  gemv_fp8.py   fp8-weight GEMV (tiled, +rmsnorm, +resid_rmsnorm; per-row/MXFP8 scale)
  gemv_fp4.py   mxfp4-weight expert GEMV (dequant, and native FP4xFP8 scaled)
  attention.py  RoPE+KV-append, per-head flash decode with sinks
  router.py     top-k + softmax
  swiglu.py     SwiGLU-OAI (fp8-quant and bf16 variants)
"""

from common.barrier import _barrier, _barrier_noinv
from common.fp4 import _fp4_lut
from common.quant import _quant_norm_fp8
from common.rmsnorm import _store_resid_rmsnorm
from common.gemv_bf16 import (
    _gemv_bf16,
    _gemv_bf16_tiled,
    _gemv_bf16_rmsnorm,
    _gemv_bf16_resid_rmsnorm,
)
from common.gemv_fp8 import (
    _gemv_fp8_tiled,
    _gemv_fp8_rmsnorm,
    _gemv_fp8_resid_rmsnorm,
)
from common.gemv_fp4 import _gemv_fp4, _gemv_fp4_scaled
from common.attention import _rope_kv_append, _flash_decode_head
from common.router import _topk_softmax
from common.swiglu import _swiglu_quant_fp8, _swiglu_bf16

__all__ = [
    "_barrier",
    "_barrier_noinv",
    "_fp4_lut",
    "_quant_norm_fp8",
    "_store_resid_rmsnorm",
    "_gemv_bf16",
    "_gemv_bf16_tiled",
    "_gemv_bf16_rmsnorm",
    "_gemv_bf16_resid_rmsnorm",
    "_gemv_fp8_tiled",
    "_gemv_fp8_rmsnorm",
    "_gemv_fp8_resid_rmsnorm",
    "_gemv_fp4",
    "_gemv_fp4_scaled",
    "_rope_kv_append",
    "_flash_decode_head",
    "_topk_softmax",
    "_swiglu_quant_fp8",
    "_swiglu_bf16",
]
