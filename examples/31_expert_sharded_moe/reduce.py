# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Expert reduce for MoE.

Matches triton_kernels/reduce.py semantics:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/reduce.py

  z[t, :] = sum_{a where mask[t,a,:]!=0} y[t, a, :]

Plain (unweighted) sum over the k expert outputs per token, gated only
by a boolean validity mask.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    Y_ptr,
    stride_y_t,
    stride_y_a,
    stride_y_d,
    Z_ptr,
    stride_z_t,
    stride_z_d,
    Mask_ptr,
    n_tokens,
    d_model,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    if pid_t >= n_tokens:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_model

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for act in range(N_EXPTS_ACT):
        if HAS_MASK:
            m = tl.load(
                Mask_ptr + pid_t * N_EXPTS_ACT * d_model + act * d_model + offs_d,
                mask=mask_d,
                other=0,
            ).to(tl.int1)
        y = tl.load(
            Y_ptr + pid_t * stride_y_t + act * stride_y_a + offs_d * stride_y_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        if HAS_MASK:
            y = tl.where(m, y, 0.0)
        acc += y

    tl.store(
        Z_ptr + pid_t * stride_z_t + offs_d * stride_z_d,
        acc.to(Z_ptr.dtype.element_ty),
        mask=mask_d,
    )


def reduce(
    y: torch.Tensor,
    dim: int = 1,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Sum-reduce over *dim* with optional boolean mask.

    Matches the upstream ``reduce(y, dim=1, mask=mask)`` signature.

    Args:
        y: (n_tokens, k, d_model) expert outputs.
        dim: reduction dimension (must be 1).
        mask: (n_tokens, k, d_model) bool/int mask; zero = skip.

    Returns:
        (z, None) where z has shape (n_tokens, d_model).
    """
    assert dim == 1 and y.ndim == 3
    n_tokens, k, d_model = y.shape
    device = y.device

    z = torch.zeros((n_tokens, d_model), dtype=y.dtype, device=device)

    BLOCK_D = min(triton.next_power_of_2(d_model), 512)
    grid = (n_tokens, triton.cdiv(d_model, BLOCK_D))

    _reduce_kernel[grid](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        z,
        z.stride(0),
        z.stride(1),
        mask if mask is not None else y,
        n_tokens,
        d_model,
        N_EXPTS_ACT=k,
        BLOCK_D=BLOCK_D,
        HAS_MASK=(mask is not None),
    )
    return z, None
