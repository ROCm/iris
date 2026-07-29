# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Auto-dispatching ReduceScatter: one-shot pull for small messages,
RCCL ring for large messages.

The physics:
  - one-shot pull: ws x data bytes over XGMI, 1 kernel launch
  - RCCL ring:     2(ws-1)/ws x data bytes, ws-1 sequential steps

Small messages: launch/step overhead dominates -> one-shot wins
Large messages: bandwidth dominates -> ring wins

Measured crossover on MI355X (ws=2, N=2880, FP16): M ~ 3000
  M=2048: one-shot 0.130ms vs RCCL 0.173ms (1.33x faster)
  M=4096: one-shot 0.314ms vs RCCL 0.308ms (0.98x, ring wins)

Crossover is expressed in bytes-per-rank so it generalizes across shapes.
"""

from typing import Optional
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris


# Crossover threshold in bytes per rank (measured on MI355X).
# Below this, one-shot pull beats RCCL ring.
#   M=2048, N=2880, fp16, ws=2 -> M_local*N*2 = 1024*2880*2 = 5.9 MB  (we win)
#   M=4096, N=2880, fp16, ws=2 -> 2048*2880*2 = 11.8 MB               (RCCL wins)
_CROSSOVER_BYTES_PER_RANK = 8 * 1024 * 1024  # 8 MB

# Per-world-size tuned configs for the one-shot kernel.
_AUTO_CONFIG = {
    2: dict(block_m=128, block_n=64, num_sms=196, num_warps=4),
    4: dict(block_m=64, block_n=64, num_sms=32, num_warps=4),
    8: dict(block_m=32, block_n=64, num_sms=32, num_warps=4),
}
_DEFAULT_CONFIG = dict(block_m=64, block_n=64, num_sms=64, num_warps=4)


@triton.jit
def _one_shot_rs_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent one-shot pull RS. Reads all peers, sums in fp32."""
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles
    m_offset = cur_rank * num_m_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + in_offset
        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (
            pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N
        )

        if is_full:
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty), mask=out_mask)


def _get_config(world_size: int, M_local: int) -> dict:
    cfg = _AUTO_CONFIG.get(world_size, _DEFAULT_CONFIG).copy()
    while cfg["block_m"] > M_local and cfg["block_m"] > 4:
        cfg["block_m"] //= 2
    return cfg


def one_shot_reduce_scatter(ctx, output_tensor, input_tensor, **kwargs):
    """
    One-shot pull RS. Input must be in the symmetric heap.

    Fastest for small messages (< ~8 MB per rank on MI355X).
    """
    M, N = input_tensor.shape
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)
    assert M % world_size == 0

    cfg = _get_config(world_size, M_local)
    bm = kwargs.get("block_m") or cfg["block_m"]
    bn = kwargs.get("block_n") or cfg["block_n"]
    sms = kwargs.get("num_sms") or cfg["num_sms"]
    warps = kwargs.get("num_warps") or cfg["num_warps"]

    assert M_local % bm == 0

    _one_shot_rs_kernel[(sms,)](
        input_tensor,
        output_tensor,
        M,
        N,
        M_local,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        ctx.get_heap_bases(),
        rank,
        world_size,
        bm,
        bn,
        sms,
        num_warps=warps,
    )


def reduce_scatter_auto(
    ctx,
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    crossover_bytes: Optional[int] = None,
    force: Optional[str] = None,
    **kwargs,
):
    """
    Auto-dispatching reduce-scatter.

    Selects one-shot pull (small messages) or RCCL ring (large messages)
    based on bytes-per-rank.

    Args:
        ctx: Iris context
        output_tensor: Output (M_local, N)
        input_tensor: Input (M, N). Must be in symmetric heap for the
            one-shot path; any tensor works for the RCCL path.
        crossover_bytes: Override the dispatch threshold (bytes per rank).
        force: "one_shot" or "rccl" to bypass auto-selection.
        **kwargs: block_m / block_n / num_sms / num_warps for the one-shot path.

    Example:
        >>> C = shmem.zeros((M, N), dtype=torch.float16)
        >>> torch.mm(A, B, out=C)
        >>> out = torch.zeros(M // ws, N, dtype=torch.float16, device="cuda")
        >>> reduce_scatter_auto(shmem, out, C)
    """
    M, N = input_tensor.shape
    world_size = ctx.get_num_ranks()
    M_local = M // world_size
    bytes_per_rank = M_local * N * input_tensor.element_size()

    threshold = crossover_bytes if crossover_bytes is not None else _CROSSOVER_BYTES_PER_RANK

    if force == "rccl":
        use_one_shot = False
    elif force == "one_shot":
        use_one_shot = True
    else:
        use_one_shot = bytes_per_rank < threshold

    if use_one_shot:
        one_shot_reduce_scatter(ctx, output_tensor, input_tensor, **kwargs)
    else:
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)


def matmul_reduce_scatter_auto(
    ctx,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    **kwargs,
):
    """
    GEMM + auto-dispatched ReduceScatter.

    Computes output = reduce_scatter(A @ B), selecting the fastest RS
    path for the message size.

    Args:
        ctx: Iris context
        output_tensor: Output (M_local, N)
        A: Input (M, K_local) — this rank's K-shard
        B: Input (K_local, N)
        **kwargs: forwarded to reduce_scatter_auto
    """
    M, K_local = A.shape
    _, N = B.shape
    world_size = ctx.get_num_ranks()
    M_local = M // world_size

    assert output_tensor.shape == (M_local, N)

    C_partial = ctx.zeros((M, N), dtype=A.dtype)
    torch.mm(A, B, out=C_partial)
    reduce_scatter_auto(ctx, output_tensor, C_partial, **kwargs)
