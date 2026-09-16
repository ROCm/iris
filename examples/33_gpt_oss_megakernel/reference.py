# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
GPT-OSS-120B batch-1 decode reference forward (PyTorch, fp32 math).

This is the numerical ground-truth spec for the single-GPU persistent Triton
megakernel in gpt_oss_120b_quantized_megakernel.py. It mirrors the HuggingFace
`GptOss` modeling math exactly:

  - RMSNorm (eps 1e-5, divisor = hidden_dim)
  - QKV projection (+bias), NeoX RoPE with YaRN scaling
  - GQA flash-style attention with per-head learned ATTENTION SINK and
    alternating sliding-window(128)/full causal masking
  - O projection (+bias) + residual
  - RMSNorm -> router (linear+bias) -> top-4 -> softmax-AFTER-topk
  - SwiGLU-OAI experts: alpha=1.702, limit=7, glu=gate*sigmoid(1.702*gate),
    out=(up+1)*glu ; gate_up & down biases ; weighted sum over the 4 experts
  - residual
  - final RMSNorm -> lm_head -> argmax (greedy)

All tensors are explicit so the same arrays feed the Triton kernels. Weights are
held as a dict (see Weights) decoupled from HF packing; load_hf() adapts a HF
GptOss state_dict into this layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class GptOssConfig:
    hidden_dim: int = 2880
    intermediate_dim: int = 2880
    num_layers: int = 36
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 201088
    num_experts: int = 128
    top_k: int = 4
    max_seq_len: int = 4096
    rope_theta: float = 150000.0
    rope_factor: float = 32.0
    rope_orig_max_pos: float = 4096.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rms_eps: float = 1e-5
    sliding_window: int = 128
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    # x: [H] fp32. RMSNorm divisor is the hidden dim, matching HF.
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * gamma.float()


def build_yarn_rope(cfg: GptOssConfig, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape [max_seq_len, head_dim/2].

    NTK-by-parts (YaRN) interpolation between
    base and scaled frequency, with an mscale=0.1*ln(factor)+1 applied to cos/sin.
    """
    DH = cfg.head_dim
    half = DH // 2
    theta = cfg.rope_theta
    factor = cfg.rope_factor
    pi = math.pi

    d = torch.arange(half, dtype=torch.float64)
    base_freq = 1.0 / (theta ** (2.0 * d / DH))
    # correction range (in dim index space)
    low = math.floor(DH * math.log(cfg.rope_orig_max_pos / (2 * pi * cfg.rope_beta_fast)) / (2 * math.log(theta)))
    high = math.ceil(DH * math.log(cfg.rope_orig_max_pos / (2 * pi * cfg.rope_beta_slow)) / (2 * math.log(theta)))
    denom = max(high - low, 1e-6)
    t = ((d - low) / denom).clamp(0.0, 1.0)
    scaled_freq = base_freq / factor
    freq = (1.0 - t) * base_freq + t * scaled_freq  # high-freq dims keep base; low-freq scaled

    mscale = 0.1 * math.log(factor) + 1.0 if factor > 0 else 1.0
    pos = torch.arange(cfg.max_seq_len, dtype=torch.float64).unsqueeze(1)  # [P,1]
    ang = pos * freq.unsqueeze(0)  # [P, half]
    cos = (torch.cos(ang) * mscale).to(torch.float32).to(device)
    sin = (torch.sin(ang) * mscale).to(torch.float32).to(device)
    return cos, sin


def apply_rope_neox(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX half-split RoPE on a single head vector.

    vec: [..., head_dim]; cos/sin: [head_dim/2]. element j pairs with j+half.
        out[j]      = x[j]*cos[j]      - x[j+half]*sin[j]
        out[j+half] = x[j+half]*cos[j] + x[j]*sin[j]
    """
    half = vec.shape[-1] // 2
    x1 = vec[..., :half]
    x2 = vec[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1)


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    # GPT-OSS SwiGLU: gate is clamped above, up is clamped both sides.
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    glu = gate * torch.sigmoid(alpha * gate)
    return (up + 1.0) * glu


class Weights:
    """Per-tensor weight container (fp32 / bf16 tensors), HF-packing-agnostic.

    Per layer l (in self.layers[l]):
      norm_attn [H], norm_moe [H]
      w_q [q_dim,H] b_q [q_dim], w_k [kv,H] b_k [kv], w_v [kv,H] b_v [kv]
      w_o [H,q_dim] b_o [H]
      sinks [num_heads]
      router_w [E,H] router_b [E]
      gate_up_w [E, 2*I, H] gate_up_b [E, 2*I]   (gate=even idx, up=odd idx interleaved)
      down_w [E, H, I] down_b [E, H]
    Global: embed [V,H], final_norm [H], lm_head [V,H]
    """

    def __init__(self):
        self.layers: list[dict] = []
        self.embed = None
        self.final_norm = None
        self.lm_head = None


@torch.no_grad()
def decode_forward(
    cfg: GptOssConfig,
    w: Weights,
    hidden: torch.Tensor,  # [H] fp32 — embedding of current token
    pos: int,  # 0-based position of current token
    kv_cache: list[dict],  # per-layer {"k":[max_seq,kv_dim], "v":[...]} fp32, k already RoPE'd
    cos: torch.Tensor,
    sin: torch.Tensor,
    capture: dict | None = None,  # optional per-phase capture for validation
) -> torch.Tensor:
    """Run one decode step. Mutates kv_cache (appends at index pos). Returns logits [V]."""
    cfg_scale = 1.0 / math.sqrt(cfg.head_dim)
    H, DH, NH, NKV = cfg.hidden_dim, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    group = NH // NKV
    cur_cos = cos[pos]  # [half]
    cur_sin = sin[pos]

    x = hidden.float()
    for l in range(cfg.num_layers):
        lw = w.layers[l]
        # --- attention input norm ---
        xn = rms_norm(x, lw["norm_attn"], cfg.rms_eps)

        q = xn @ lw["w_q"].float().T + lw["b_q"].float()  # [q_dim]
        k = xn @ lw["w_k"].float().T + lw["b_k"].float()  # [kv_dim]
        v = xn @ lw["w_v"].float().T + lw["b_v"].float()  # [kv_dim]

        q = q.view(NH, DH)
        k = k.view(NKV, DH)
        v = v.view(NKV, DH)
        # RoPE on q,k (NeoX)
        q = apply_rope_neox(q, cur_cos, cur_sin)
        k = apply_rope_neox(k, cur_cos, cur_sin)

        # append to cache at pos
        kv_cache[l]["k"][pos] = k.reshape(-1)
        kv_cache[l]["v"][pos] = v.reshape(-1)

        window = cfg.sliding_window if (l % 2 == 0) else 0
        # valid key positions: 0..pos ; sliding window uses strict (pos - j) < window
        if window > 0:
            lo = max(0, pos - window + 1)
        else:
            lo = 0
        Kc = kv_cache[l]["k"][lo : pos + 1].view(-1, NKV, DH).float()  # [T, NKV, DH]
        Vc = kv_cache[l]["v"][lo : pos + 1].view(-1, NKV, DH).float()
        T = Kc.shape[0]

        attn_out = torch.empty(NH, DH, dtype=torch.float32, device=x.device)
        for h in range(NH):
            kvh = h // group
            kh = Kc[:, kvh, :]  # [T, DH]
            vh = Vc[:, kvh, :]
            scores = (kh @ q[h]) * cfg_scale  # [T]
            sink = lw["sinks"][h].float()
            m = torch.max(scores.max(), sink)
            e = torch.exp(scores - m)
            denom = e.sum() + torch.exp(sink - m)  # sink in denominator
            p = e / denom
            attn_out[h] = (p.unsqueeze(1) * vh).sum(dim=0)

        attn_flat = attn_out.reshape(-1)  # [q_dim]
        o = attn_flat @ lw["w_o"].float().T + lw["b_o"].float()  # [H]
        x = x + o  # residual

        if capture is not None and l == capture.get("layer", -1):
            capture["attn_out"] = attn_flat.clone()
            capture["post_attn"] = x.clone()

        # --- MoE ---
        xn2 = rms_norm(x, lw["norm_moe"], cfg.rms_eps)
        router_logits = xn2 @ lw["router_w"].float().T + lw["router_b"].float()  # [E]
        topv, topi = torch.topk(router_logits, cfg.top_k)
        gate_w = torch.softmax(topv, dim=0)  # softmax AFTER topk

        moe_acc = torch.zeros(H, dtype=torch.float32, device=x.device)
        I = cfg.intermediate_dim
        # experts kept MXFP4 (load_hf) or pre-dequantized (random); support both.
        have_fp4 = "gate_up_blocks" in lw
        for slot in range(cfg.top_k):
            e_id = int(topi[slot])
            gw = gate_w[slot]
            if have_fp4:
                from load_hf import dequant_mxfp4_rows

                gu_w = dequant_mxfp4_rows(lw["gate_up_blocks"][e_id], lw["gate_up_scales"][e_id])  # [2I,H]
                dw = dequant_mxfp4_rows(lw["down_blocks"][e_id], lw["down_scales"][e_id])  # [H,I]
            else:
                gu_w = lw["gate_up_w"][e_id].float()  # [2I, H]
                dw = lw["down_w"][e_id].float()  # [H, I]
            gu_b = lw["gate_up_b"][e_id].float()  # [2I]
            gu = xn2 @ gu_w.T + gu_b  # [2I]
            gate = gu[0::2]  # [I]
            up = gu[1::2]  # [I]
            act = swiglu_oai(gate, up, cfg.swiglu_alpha, cfg.swiglu_limit)  # [I]
            db = lw["down_b"][e_id].float()  # [H]
            ev = act @ dw.T + db  # [H]
            moe_acc += gw * ev

        x = x + moe_acc

        if capture is not None and l == capture.get("layer", -1):
            capture["moe_out"] = moe_acc.clone()
            capture["post_moe"] = x.clone()

    xf = rms_norm(x, w.final_norm, cfg.rms_eps)
    logits = xf @ w.lm_head.float().T  # [V]
    return logits
