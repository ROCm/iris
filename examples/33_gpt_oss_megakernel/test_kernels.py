# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Validate each Triton phase kernel against the PyTorch reference math."""

from __future__ import annotations

import torch

import kernels as K
from reference import swiglu_oai, apply_rope_neox, rms_norm

dev = "cuda"
torch.manual_seed(0)


def report(name, a, b, atol=1e-2, rtol=1e-2):
    a = a.float().flatten()
    b = b.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)
    maxerr = (a - b).abs().max().item()
    ok = maxerr < atol + rtol * b.abs().max().item()
    print(f"[{'OK ' if ok else 'BAD'}] {name:18s} cos={cos:.6f} maxerr={maxerr:.4e}")
    return ok


def test_rmsnorm():
    H = 2880
    x = torch.randn(H, device=dev)
    g = torch.randn(H, device=dev)
    out = K.rmsnorm(x, g, 1e-5)
    ref = rms_norm(x, g, 1e-5)
    report("rmsnorm", out, ref)


def test_gemv():
    M, Kd = 4096, 2880
    W = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
    x = torch.randn(Kd, device=dev)
    b = torch.randn(M, device=dev, dtype=torch.bfloat16)
    out = K.gemv(W, x, b)
    ref = x @ W.float().T + b.float()
    report("gemv", out, ref)


def test_swiglu():
    I = 2880
    gu = torch.randn(2 * I, device=dev) * 3
    out = K.swiglu(gu, I, 1.702, 7.0)
    ref = swiglu_oai(gu[0::2], gu[1::2], 1.702, 7.0)
    report("swiglu", out, ref)


def test_rope():
    NH, NKV, DH = 64, 8, 64
    q = torch.randn(NH * DH, device=dev)
    k = torch.randn(NKV * DH, device=dev)
    v = torch.randn(NKV * DH, device=dev)
    half = DH // 2
    cos = torch.randn(half, device=dev)
    sin = torch.randn(half, device=dev)
    kc = torch.zeros(16, NKV * DH, device=dev)
    vc = torch.zeros(16, NKV * DH, device=dev)
    q2 = q.clone()
    K.rope_and_cache(q2, k.clone(), v.clone(), cos, sin, kc, vc, 3, NH, NKV, DH)
    # ref
    qr = apply_rope_neox(q.view(NH, DH), cos, sin).reshape(-1)
    kr = apply_rope_neox(k.view(NKV, DH), cos, sin).reshape(-1)
    report("rope_q", q2, qr)
    report("rope_k_cache", kc[3], kr)
    report("v_cache", vc[3], v)


def test_attention():
    NH, NKV, DH = 64, 8, 64
    pos = 10
    kv_dim = NKV * DH
    q = torch.randn(NH * DH, device=dev)
    kc = torch.randn(16, kv_dim, device=dev)
    vc = torch.randn(16, kv_dim, device=dev)
    sinks = torch.randn(NH, device=dev)
    scale = 1.0 / (DH**0.5)
    window = 0
    out = K.attention(q, kc, vc, sinks, pos, window, scale, NH, NKV, DH)
    # ref
    group = NH // NKV
    qv = q.view(NH, DH)
    ref = torch.empty(NH, DH, device=dev)
    for h in range(NH):
        kvh = h // group
        kh = kc[: pos + 1, kvh * DH : (kvh + 1) * DH]
        vh = vc[: pos + 1, kvh * DH : (kvh + 1) * DH]
        s = (kh @ qv[h]) * scale
        sink = sinks[h]
        m = torch.max(s.max(), sink)
        e = torch.exp(s - m)
        den = e.sum() + torch.exp(sink - m)
        ref[h] = (e.unsqueeze(1) * vh).sum(0) / den
    report("attention", out, ref.reshape(-1))


def test_dequant():
    # build a random fp4 expert row block and compare to load_hf.dequant_mxfp4_rows
    from load_hf import dequant_mxfp4_rows

    R, nblk = 8, 90
    blocks = torch.randint(0, 256, (R, nblk, 16), dtype=torch.uint8, device=dev)
    scales = torch.randint(120, 130, (R, nblk), dtype=torch.uint8, device=dev)
    out = K.dequant_fp4(blocks, scales)
    ref = dequant_mxfp4_rows(blocks, scales)
    report("dequant_fp4", out, ref, atol=1e-3)


def test_argmax():
    x = torch.randn(201088, device=dev)
    idx = K.argmax(x)
    ref = int(torch.argmax(x))
    print(f"[{'OK ' if idx == ref else 'BAD'}] argmax            kernel={idx} ref={ref}")


if __name__ == "__main__":
    test_rmsnorm()
    test_gemv()
    test_swiglu()
    test_rope()
    test_attention()
    test_dequant()
    test_argmax()
