# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Test the quantized expert GEMV against a PyTorch reference.

Compares both expert paths -- the FP4 x FP8 scaled multiply and the BF16 dequant
multiply -- on the real expert shapes (gate-up: N = 2*I, K = H; down: N = H,
K = I), to confirm the scaled GEMV layout and the activation quantization."""

from __future__ import annotations

import torch
import triton

import gpt_oss_120b_quantized_megakernel as MK
import load_hf

dev = "cuda"
NUM_WG = MK.NUM_WG
torch.manual_seed(0)


def ref_dequant_gemv(blk, scl, x, bias):
    # blk [N, nb, 16] uint8, scl [N, nb], x [K] fp32 -> y [N]
    W = load_hf.dequant_mxfp4_rows(blk, scl)  # [N, K]
    return W @ x.float() + bias.float()


def run_bf16(blk, scl, x, bias, N, K, NB):
    # _gemv_fp4 expects blk flat [N, K//2], scl [N, NB], x [K]
    blkf = blk.reshape(N, -1).contiguous()
    y = torch.zeros(N, dtype=torch.float32, device=dev)
    grid = (NUM_WG,)

    import triton.language as tl

    @triton.jit
    def _run(blk_base, scl_base, x_ptr, y_ptr, b_base, N, K: tl.constexpr, NB: tl.constexpr):
        pid = tl.program_id(0)
        MK._gemv_fp4(blk_base, scl_base, x_ptr, y_ptr, b_base, True, N, K, NB, pid, 1.0, ACCUM=False)

    _run[grid](blkf, scl, x, y, bias, N, K, NB, num_warps=4)
    return y


def run_quant(blk, scl, x, bias, N, K, NB):
    blkf = blk.reshape(N, -1).contiguous()
    # quant activation
    import triton.language as tl

    afp8 = torch.zeros(K, dtype=torch.float8_e4m3fn, device=dev)
    ascl = torch.zeros(NB, dtype=torch.uint8, device=dev)

    @triton.jit
    def _q(x_ptr, fp8_ptr, scl_ptr, K, NB: tl.constexpr):
        pid = tl.program_id(0)
        MK._quant_act_fp8(x_ptr, fp8_ptr, scl_ptr, K, NB, pid)

    _q[(NUM_WG,)](x, afp8, ascl, K, NB, num_warps=4)

    y = torch.zeros(N, dtype=torch.float32, device=dev)

    @triton.jit
    def _g(
        blk_base,
        scl_base,
        afp8_ptr,
        ascl_ptr,
        y_ptr,
        b_base,
        N,
        K: tl.constexpr,
        NB: tl.constexpr,
        BLOCK_NQ: tl.constexpr,
        BLOCK_KQ: tl.constexpr,
        MTILE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        MK._gemv_fp4_scaled(
            blk_base,
            scl_base,
            afp8_ptr,
            ascl_ptr,
            y_ptr,
            b_base,
            True,
            N,
            K,
            NB,
            pid,
            1.0,
            False,
            BLOCK_NQ,
            BLOCK_KQ,
            MTILE,
        )

    _g[(NUM_WG,)](blkf, scl, afp8, ascl, y, bias, N, K, NB, BLOCK_NQ=64, BLOCK_KQ=128, MTILE=16, num_warps=4)
    return y, afp8, ascl


def cmp(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)
    print(f"{name:24s} cos={cos:.6f} maxerr={(a-b).abs().max():.4e}")


def test_shape(N, K, tag):
    NB = K // 32
    blk = torch.randint(0, 256, (N, NB, 16), dtype=torch.uint8, device=dev)
    scl = torch.randint(124, 130, (N, NB), dtype=torch.uint8, device=dev)
    x = torch.randn(K, device=dev)
    bias = torch.randn(N, device=dev, dtype=torch.bfloat16)
    ref = ref_dequant_gemv(blk, scl, x, bias)
    ybf = run_bf16(blk, scl, x, bias, N, K, NB)
    yq, afp8, ascl = run_quant(blk, scl, x, bias, N, K, NB)
    print(f"--- {tag} N={N} K={K} ---")
    cmp("bf16 vs ref", ybf, ref)
    cmp("quant vs ref", yq, ref)
    cmp("quant vs bf16", yq, ybf)
    # also: torch-side dot_scaled-equivalent (dequant act the same way)
    a_deq = afp8.float() * torch.repeat_interleave(
        torch.where(ascl > 0, torch.exp2(ascl.float() - 127), torch.ones_like(ascl.float())), 32
    )
    W = load_hf.dequant_mxfp4_rows(blk, scl)
    ref_q = W @ a_deq + bias.float()
    cmp("quant vs ref_q(act)", yq, ref_q)
    print("  yq[:4] ", yq[:4].tolist())
    print("  refq[:4]", ref_q[:4].tolist())


if __name__ == "__main__":
    test_shape(2 * 2880, 2880, "gate_up")
    test_shape(2880, 2880, "down")
