# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Validate tl.dot_scaled (MXFP4 e2m1 weights x FP8 e4m3 activations, e8m0 per-32
scales) for a batch-1 GEMV, against the BF16 dequant reference. Establishes the
exact operand layout before reworking the megakernel.

Problem: y[n] = sum_k W[n,k] * a[k], W is FP4 [N,K] row-major (HF layout),
a is the activation vector [K]. dot_scaled computes lhs[M,K] @ rhs[K,N] -> [M,N].

We map:  lhs = activation tile  [M, K]  e4m3   (M = MFMA tile, row 0 = real token)
         rhs = weight^T        [K, N]  e2m1   (so column n = W[n, :])
         out = [M, N], take row 0 -> y[n] for n in this N-tile.

FP4 packing: e2m1 nibbles packed 2 per uint8 along K (low nibble first). Our HF
weight blocks already are [N, K//2] uint8 with low=even, high=odd along K, which
is exactly K-packed for the [.,K] dimension. For rhs we need [K, N] with K packed
-> rhs is W^T packed along K: shape [K//2, N] uint8.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---- FP8 e4m3 activation quant (per-32 e8m0, amax/448) ----
def quant_e4m3_e8m0(a: torch.Tensor, group=32):
    # a: [K] fp32 -> (fp8_uint8 [K], scale_e8m0 [K//32])
    K = a.numel()
    ab = a.view(K // group, group)
    amax = ab.abs().amax(dim=1)  # [K/32]
    # e8m0: raw_exp of amax/448, round up if mantissa nonzero
    target = (amax / 448.0).float()
    u = target.view(torch.int32)
    raw = (u >> 23) & 0xFF
    raw = raw + ((u & 0x7FFFFF) != 0).int()
    raw = torch.clamp(raw, 0, 255).to(torch.uint8)
    raw = torch.where(amax == 0, torch.zeros_like(raw), raw)
    scale = torch.where(raw > 0, torch.exp2(raw.float() - 127.0), torch.ones_like(amax))
    aq = (ab / scale.unsqueeze(1)).reshape(K)
    # cast to e4m3 (torch float8_e4m3fn) then bitcast to uint8
    fp8 = aq.to(torch.float8_e4m3fn)
    return fp8, raw  # fp8 tensor (K,), raw e8m0 (K/32,)


@triton.jit
def _dot_scaled_gemv(
    wq_ptr,  # weight FP4 packed [N, K//2] uint8 (row-major, low nibble=even k)
    wscl_ptr,  # weight scales e8m0 [N, K//32] uint8
    a_ptr,  # activation FP8 e4m3 [K] (as uint8/float8)
    ascl_ptr,  # activation scales e8m0 [K//32] uint8
    y_ptr,  # out [N] fp32
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    rowsM = tl.arange(0, M)
    SB: tl.constexpr = BLOCK_K // 32
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        kmask = kk < K
        kp = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        kpmask = kp < (K // 2)
        sb = (k0 // 32) + tl.arange(0, SB)
        sbmask = sb < (K // 32)
        # lhs = activation [M, BLOCK_K] e4m3, only row 0 real (masked tail -> 0)
        a = tl.load(a_ptr + kk[None, :], mask=(rowsM[:, None] == 0) & kmask[None, :], other=0).to(tl.uint8)
        ascl = tl.load(ascl_ptr + sb[None, :], mask=sbmask[None, :], other=0)
        ascl = tl.broadcast_to(ascl, (M, SB))
        # rhs = W^T : [BLOCK_K//2, BLOCK_N] packed along K  (wq[n, kp])
        w = tl.load(wq_ptr + n[None, :] * (K // 2) + kp[:, None], mask=nmask[None, :] & kpmask[:, None], other=0).to(
            tl.uint8
        )
        wscl = tl.load(wscl_ptr + n[:, None] * (K // 32) + sb[None, :], mask=nmask[:, None] & sbmask[None, :], other=0)
        acc = tl.dot_scaled(a, ascl, "e4m3", w, wscl, "e2m1", acc=acc, out_dtype=tl.float32)
    y = tl.sum(tl.where(rowsM[:, None] == 0, acc, 0.0), axis=0)
    tl.store(y_ptr + n, y, mask=nmask)


def dot_scaled_gemv(wq, wscl, a_fp8, a_scl, N, K, BLOCK_N=64, BLOCK_K=128, M=16):
    y = torch.empty(N, dtype=torch.float32, device=wq.device)
    grid = (triton.cdiv(N, BLOCK_N),)
    _dot_scaled_gemv[grid](
        wq, wscl, a_fp8.view(torch.uint8), a_scl, y, N, K, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, M=M, num_warps=4
    )
    return y


def main():
    import load_hf

    dev = "cuda"
    torch.manual_seed(0)
    N, K = 256, 2880
    # random FP4 weight in HF block layout [N, K//32, 16] + scales [N, K//32]
    blk = torch.randint(0, 256, (N, K // 32, 16), dtype=torch.uint8, device=dev)
    scl = torch.randint(124, 130, (N, K // 32), dtype=torch.uint8, device=dev)
    Wf = load_hf.dequant_mxfp4_rows(blk, scl)  # [N,K] fp32 reference weight
    wq = blk.reshape(N, K // 2).contiguous()  # [N, K//2] packed along K

    a = torch.randn(K, device=dev)
    a_fp8, a_scl = quant_e4m3_e8m0(a)

    y = dot_scaled_gemv(wq, scl, a_fp8, a_scl, N, K)
    # reference: dequant both sides the same way the hw would
    a_deq = a_fp8.float() * torch.repeat_interleave(
        torch.where(a_scl > 0, torch.exp2(a_scl.float() - 127.0), torch.ones_like(a_scl.float())), 32
    )
    ref = Wf @ a_deq
    cos = torch.dot(y, ref) / (y.norm() * ref.norm() + 1e-9)
    print(f"dot_scaled GEMV  cos={cos:.6f}  maxerr={(y-ref).abs().max():.4e}")
    print("y[:5]  ", y[:5].tolist())
    print("ref[:5]", ref[:5].tolist())


if __name__ == "__main__":
    main()
