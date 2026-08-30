# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Load an OpenAI GPT-OSS-120B HuggingFace checkpoint into the reference Weights
layout. Experts are stored MXFP4 in HF (E2M1 nibble blocks + E8M0 per-32 scales,
gate/up interleaved); everything else is BF16/FP32. For the BF16-first kernel we
dequantize experts to fp32/bf16 on load.

HF tensor names (per layer l):
  model.layers.{l}.input_layernorm.weight                      [H] bf16
  model.layers.{l}.self_attn.{q,k,v,o}_proj.{weight,bias}
  model.layers.{l}.self_attn.sinks                             [num_heads]
  model.layers.{l}.post_attention_layernorm.weight             [H]
  model.layers.{l}.mlp.router.{weight,bias}                    [E,H],[E]
  model.layers.{l}.mlp.experts.gate_up_proj_blocks  [E, 2I, H/2] uint8 (nibbles)
  model.layers.{l}.mlp.experts.gate_up_proj_scales  [E, 2I, H/32] uint8 (E8M0)
  model.layers.{l}.mlp.experts.gate_up_proj_bias    [E, 2I]   (interleaved gate,up)
  model.layers.{l}.mlp.experts.down_proj_blocks     [E, H, I/2]
  model.layers.{l}.mlp.experts.down_proj_scales     [E, H, I/32]
  model.layers.{l}.mlp.experts.down_proj_bias       [E, H]
  model.embed_tokens.weight [V,H]; model.norm.weight [H]; lm_head.weight [V,H]

Note on expert weight orientation: HF stores gate_up_proj as [E, 2I, H] (out=2I,
in=H) and down_proj as [E, H, I] (out=H, in=I) AFTER dequant. The blocks tensors
nibble-pack the *input* dim, so gate_up_proj_blocks is [E, 2I, H/2].
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from reference import GptOssConfig, Weights

_FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _find_snapshot(repo_cache: str | None = None) -> str:
    if repo_cache and os.path.isdir(repo_cache):
        cand = glob.glob(os.path.join(repo_cache, "snapshots", "*"))
        if cand:
            return cand[0]
        return repo_cache
    home = os.path.expanduser("~")
    base = os.path.join(home, ".cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots")
    cand = sorted(glob.glob(os.path.join(base, "*")))
    if not cand:
        raise FileNotFoundError(f"No gpt-oss-120b snapshot under {base}")
    return cand[-1]


_FP4_LUT_T = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """blocks [..., C/2] uint8 nibble pairs, scales [..., C/32] uint8 E8M0.
    Returns float32 [..., C]. low nibble = even col, high nibble = odd col."""
    blocks = np.ascontiguousarray(blocks).astype(np.uint8)
    scales = np.ascontiguousarray(scales).astype(np.uint8)
    *lead, half = blocks.shape
    C = half * 2
    lo = blocks & 0x0F
    hi = (blocks >> 4) & 0x0F

    def lut(n):
        mag = _FP4_LUT[n & 7]
        return np.where((n & 8) != 0, -mag, mag)

    vlo = lut(lo)  # [..., C/2] even cols
    vhi = lut(hi)  # [..., C/2] odd cols
    out = np.empty((*lead, C), dtype=np.float32)
    out[..., 0::2] = vlo
    out[..., 1::2] = vhi
    # scale: 2^(e-127), e==0 -> 0
    sc = np.where(scales > 0, np.exp2(scales.astype(np.float32) - 127.0), 0.0).astype(np.float32)
    sc = np.repeat(sc, 32, axis=-1)  # [..., C]
    return out * sc


def dequant_mxfp4_rows(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Torch, single expert. Dequant only routed experts on the fly (memory-safe).

    HF layout: blocks [R, n_blocks, 16] uint8 (each 16 bytes = 32 nibble values for
    one 32-element scale block), scales [R, n_blocks] uint8 E8M0. Returns fp32 [R, C]
    with C = n_blocks*32. Within a block: byte b -> elem 2b (low nibble), 2b+1 (high)."""
    dev = blocks.device
    lut = _FP4_LUT_T.to(dev)
    if blocks.dim() == 3:
        R, nb, _ = blocks.shape
        b = blocks.reshape(R, nb * 16).to(torch.int64)  # [R, C/2]
    else:
        R, half = blocks.shape
        nb = half // 16
        b = blocks.to(torch.int64)
    C = b.shape[1] * 2
    lo = lut[b & 0xF]  # [R, C/2] even cols
    hi = lut[(b >> 4) & 0xF]  # [R, C/2] odd cols
    out = torch.empty(R, C, dtype=torch.float32, device=dev)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    s = scales.reshape(R, nb).to(torch.float32)
    sc = torch.where(s > 0, torch.exp2(s - 127.0), torch.zeros_like(s))  # [R, n_blocks]
    sc = sc.repeat_interleave(32, dim=1)  # [R, C]
    return out * sc


@torch.no_grad()
def load_hf_weights(
    cfg: GptOssConfig, snapshot: str | None = None, num_layers: int | None = None, device="cpu", dtype=torch.float32
) -> Weights:
    from safetensors import safe_open

    snap = _find_snapshot(snapshot)
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    # group keys by file to open each shard once
    file_to_keys: dict[str, list[str]] = {}
    for k, fn in weight_map.items():
        file_to_keys.setdefault(fn, []).append(k)

    cache: dict[str, torch.Tensor] = {}
    L = num_layers if num_layers is not None else cfg.num_layers
    needed_prefixes = tuple(f"model.layers.{l}." for l in range(L))

    def want(k: str) -> bool:
        if k.startswith("model.layers."):
            return k.startswith(needed_prefixes)
        return True  # embed, norm, lm_head

    for fn, keys in file_to_keys.items():
        path = os.path.join(snap, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                if want(k):
                    cache[k] = f.get_tensor(k)

    def g(name) -> torch.Tensor:
        return cache[name]

    w = Weights()
    w.embed = g("model.embed_tokens.weight").to(dtype)
    w.final_norm = g("model.norm.weight").float()
    w.lm_head = g("lm_head.weight").to(dtype)

    for l in range(L):
        p = f"model.layers.{l}."
        lw = {}
        lw["norm_attn"] = g(p + "input_layernorm.weight").float()
        lw["norm_moe"] = g(p + "post_attention_layernorm.weight").float()
        lw["w_q"] = g(p + "self_attn.q_proj.weight").to(dtype)
        lw["b_q"] = g(p + "self_attn.q_proj.bias").to(dtype)
        lw["w_k"] = g(p + "self_attn.k_proj.weight").to(dtype)
        lw["b_k"] = g(p + "self_attn.k_proj.bias").to(dtype)
        lw["w_v"] = g(p + "self_attn.v_proj.weight").to(dtype)
        lw["b_v"] = g(p + "self_attn.v_proj.bias").to(dtype)
        lw["w_o"] = g(p + "self_attn.o_proj.weight").to(dtype)
        lw["b_o"] = g(p + "self_attn.o_proj.bias").to(dtype)
        lw["sinks"] = g(p + "self_attn.sinks").float()
        lw["router_w"] = g(p + "mlp.router.weight").to(dtype)
        lw["router_b"] = g(p + "mlp.router.bias").to(dtype)

        # experts: KEEP MXFP4 (memory-safe). Dequant only routed experts at runtime.
        lw["gate_up_blocks"] = g(p + "mlp.experts.gate_up_proj_blocks")  # [E,2I,H/2] uint8
        lw["gate_up_scales"] = g(p + "mlp.experts.gate_up_proj_scales")  # [E,2I,H/32] uint8
        lw["gate_up_b"] = g(p + "mlp.experts.gate_up_proj_bias").to(dtype)  # [E,2I] interleaved
        lw["down_blocks"] = g(p + "mlp.experts.down_proj_blocks")  # [E,H,I/2] uint8
        lw["down_scales"] = g(p + "mlp.experts.down_proj_scales")  # [E,H,I/32] uint8
        lw["down_b"] = g(p + "mlp.experts.down_proj_bias").to(dtype)  # [E,H]

        w.layers.append(lw)

    return w
