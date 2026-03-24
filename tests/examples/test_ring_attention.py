################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Correctness tests for Ring Attention.
#
# Each test validates the distributed ring-attention output against a
# single-device PyTorch reference implementation.
#
################################################################################

import gc
import sys
from pathlib import Path

import pytest
import torch
import iris

project_root = Path(__file__).resolve()
while not (project_root / "tests").is_dir() or not (project_root / "examples").is_dir():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent

module_dir = project_root / "examples" / "32_ring_attention"
if module_dir.exists():
    sys.path.insert(0, str(module_dir))

from ring_attention_layer import RingAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Reference (single-device) implementation
# ---------------------------------------------------------------------------


def ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """
    Reference causal/non-causal self-attention on a single device.

    Args:
        q: ``[total_seq, num_heads, head_dim]``
        k: ``[total_seq, num_heads, head_dim]``
        v: ``[total_seq, num_heads, head_dim]``
        scale: Softmax scale factor.
        causal: Whether to apply causal masking.

    Returns:
        Attention output ``[total_seq, num_heads, head_dim]``.
    """
    total_seq, num_heads, head_dim = q.shape
    # Work in float32 for reference accuracy
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # [num_heads, total_seq, head_dim]
    q_h = q_f.permute(1, 0, 2)
    k_h = k_f.permute(1, 0, 2)
    v_h = v_f.permute(1, 0, 2)

    # Attention scores: [num_heads, total_seq, total_seq]
    attn = torch.bmm(q_h, k_h.transpose(-1, -2)) * scale

    if causal:
        mask = torch.triu(torch.ones(total_seq, total_seq, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    out = torch.bmm(attn, v_h)  # [num_heads, total_seq, head_dim]
    return out.permute(1, 0, 2).to(q.dtype)  # [total_seq, num_heads, head_dim]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_ring_attn_test(total_seq_len, num_heads, head_dim, causal, dtype):
    """Run one correctness check; called from test functions."""
    shmem = None
    try:
        shmem = iris.iris()
        rank = shmem.get_rank()
        num_ranks = shmem.get_num_ranks()

        torch.set_default_device("cuda")
        torch.manual_seed(0)

        scale = head_dim**-0.5
        seq_local = total_seq_len // num_ranks

        # Rank 0 creates the full Q, K, V and broadcasts to all ranks so that
        # the reference and distributed implementations see the same data.
        if rank == 0:
            q_full = torch.randn(total_seq_len, num_heads, head_dim, dtype=dtype) * 0.1
            k_full = torch.randn(total_seq_len, num_heads, head_dim, dtype=dtype) * 0.1
            v_full = torch.randn(total_seq_len, num_heads, head_dim, dtype=dtype) * 0.1
        else:
            q_full = torch.empty(total_seq_len, num_heads, head_dim, dtype=dtype)
            k_full = torch.empty(total_seq_len, num_heads, head_dim, dtype=dtype)
            v_full = torch.empty(total_seq_len, num_heads, head_dim, dtype=dtype)

        q_full = torch.from_numpy(shmem.broadcast(q_full.cpu().numpy(), source_rank=0)).to(q_full.device)
        k_full = torch.from_numpy(shmem.broadcast(k_full.cpu().numpy(), source_rank=0)).to(k_full.device)
        v_full = torch.from_numpy(shmem.broadcast(v_full.cpu().numpy(), source_rank=0)).to(v_full.device)

        # Local chunks for this rank
        q_local = q_full[rank * seq_local : (rank + 1) * seq_local].contiguous()
        k_local = k_full[rank * seq_local : (rank + 1) * seq_local].contiguous()
        v_local = v_full[rank * seq_local : (rank + 1) * seq_local].contiguous()

        shmem.barrier()

        # --- Distributed ring attention ---
        layer = RingAttention(shmem, num_heads=num_heads, head_dim=head_dim, causal=causal, scale=scale)
        output_local = layer(q_local, k_local, v_local)
        torch.cuda.synchronize()

        # --- Single-device reference ---
        ref_full = ref_attention(q_full, k_full, v_full, scale=scale, causal=causal)
        ref_local = ref_full[rank * seq_local : (rank + 1) * seq_local]

        shmem.barrier()

        # Tolerances: FP16 MFMA accumulation + online softmax across multiple
        # ring steps introduces rounding that grows with world_size.
        atol, rtol = (3e-2, 3e-2) if dtype == torch.float16 else (1e-2, 1e-2)
        error = None
        try:
            torch.testing.assert_close(output_local.float(), ref_local.float(), atol=atol, rtol=rtol)
        except AssertionError as e:
            error = e

        # Print a brief report from rank 0
        if rank == 0:
            max_diff = (output_local.float() - ref_local.float()).abs().max().item()
            status = "PASSED" if error is None else "FAILED"
            print(
                f"[Rank 0] Ring Attention test {status} | "
                f"seq={total_seq_len} h={num_heads} d={head_dim} "
                f"causal={causal} dtype={dtype} | max_diff={max_diff:.6f}"
            )

        shmem.barrier()

        if error is not None:
            raise error

    finally:
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass
            del shmem
            gc.collect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("total_seq_len", [512, 2048])
@pytest.mark.parametrize("causal", [True, False])
def test_ring_attention_correctness(total_seq_len, num_heads, head_dim, causal):
    """
    Validate ring attention output against a single-device PyTorch reference
    for both causal and bidirectional modes.
    """
    _run_ring_attn_test(
        total_seq_len=total_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype=torch.float16,
    )
