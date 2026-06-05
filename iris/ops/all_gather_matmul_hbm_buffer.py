# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM using a local HBM staging buffer with dedicated
fetcher and GEMM workgroups, launched data-parallel.

Supports configurable staged_a buffer layout (M-contiguous or K-contiguous)
and B layout to match optimal tritonblas conventions (TN, TT, NT, NN).
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from iris.host.tracing.events import TraceEvent
from .config import FusedConfig
from .workspace import FusedWorkspace


# ──────────────────────────────────────────────────────────────────────
# Auto-config: shape-adaptive parameter selection for HBM buffer kernel
# Source: K-021 sweep data (1076+ trials, 7 verified champion shapes)
# ──────────────────────────────────────────────────────────────────────

# Verified champion configs from IRIS-0018/0019 sweeps + optimize-loop iter3.
# Key: (M, N, K) -> dict of kernel params that beat PyTorch.
_CHAMPION_CONFIGS = {
    (262144, 8192, 8192): dict(
        bm=256,
        bn=256,
        bk=64,
        gm=24,
        kpf=64,
        fs=52,
        nfs=128,
        fsf=304,
    ),
    (131072, 16384, 16384): dict(
        bm=256,
        bn=256,
        bk=64,
        gm=24,
        kpf=32,
        fs=4,
        nfs=64,
        fsf=52,
    ),
    (147456, 28672, 4096): dict(
        bm=256,
        bn=256,
        bk=64,
        gm=24,
        kpf=16,
        fs=59,
        nfs=36,
        fsf=52,
    ),
    (229376, 28672, 4096): dict(
        bm=256,
        bn=256,
        bk=64,
        gm=24,
        kpf=16,
        fs=4,
        nfs=56,
        fsf=52,
    ),
    (327680, 28672, 4096): dict(
        bm=256,
        bn=256,
        bk=64,
        gm=24,
        kpf=16,
        fs=4,
        nfs=32,
        fsf=52,
    ),
    (8192, 8192, 262144): dict(
        bm=128,
        bn=256,
        bk=64,
        gm=8,
        kpf=32,
        fs=4,
        nfs=8,
        fsf=52,
    ),
    (16384, 16384, 131072): dict(
        bm=128,
        bn=256,
        bk=64,
        gm=16,
        kpf=16,
        fs=16,
        nfs=8,
        fsf=52,
    ),
}


def _auto_config(M: int, N: int, K: int, world_size: int = 8):
    """
    Select optimal HBM buffer kernel parameters for a given shape.

    Returns (FusedConfig, k_per_flag, num_fetch_sms, num_fetch_stages,
             first_stage_fetch_sms) — ready to pass to the kernel.

    Priority order:
      1. Exact match in champion configs (verified 1.12-1.44x vs PyTorch)
      2. Shape-heuristic derivation from 1076-trial sweep principles

    Heuristics (from K-021 sweep analysis):
      - k_per_flag is the #1 knob (52% of perf range). Maximize it.
      - bm=256 for M%256==0 and M>=8K; bm=128 otherwise
      - bn=256 always (bn=128 is 15-35% worse)
      - bk=64 always (bk=128 exceeds 64KB LDS on MI300X)
      - num_stages=2 always (num_stages=3 crashes — 98KB LDS needed)
      - num_warps=8 always (fewer warps = 22% worse)
      - group_size_m: 1 for small M, 24 for large M (L2 locality)
    """
    key = (M, N, K)
    if key in _CHAMPION_CONFIGS:
        c = _CHAMPION_CONFIGS[key]
        # Validate kpf for this world_size
        num_k_blocks = K // c["bk"]
        kpf = c["kpf"]
        while num_k_blocks % kpf != 0 and kpf > 1:
            kpf //= 2
        config = FusedConfig(
            block_size_m=c["bm"],
            block_size_n=c["bn"],
            block_size_k=c["bk"],
            group_size_m=c["gm"],
        )
        return config, kpf, c["fs"], c["nfs"], c["fsf"]

    # Derive from heuristics
    num_k_blocks = K // 64

    # Block sizes
    bm = 256 if (M % 256 == 0 and M >= 8192) else 128
    num_m_tiles = M // bm

    # k_per_flag: maximize for throughput
    if num_k_blocks >= 512:
        kpf = 64
    elif num_k_blocks >= 128:
        kpf = 16
    elif num_k_blocks >= 64:
        kpf = 8
    else:
        kpf = 4
    while num_k_blocks % kpf != 0 and kpf > 1:
        kpf //= 2

    # num_fetch_sms: scale with M-tiles (more tiles → more fetchers)
    if num_m_tiles <= 8:
        fs = 4
    elif num_m_tiles <= 32:
        fs = 16
    elif num_m_tiles <= 128:
        fs = 32
    else:
        fs = 52

    # num_fetch_stages
    if num_m_tiles >= 512:
        nfs = 4
    elif num_m_tiles >= 64:
        nfs = 2
    else:
        nfs = 1

    # group_size_m
    gm = 24 if num_m_tiles >= 64 else (8 if num_m_tiles >= 16 else 1)

    config = FusedConfig(
        block_size_m=bm,
        block_size_n=256,
        block_size_k=64,
        group_size_m=gm,
    )
    return config, kpf, fs, nfs, 64


@triton.jit
def _hbm_buffer_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
    M,
    N,
    K,
    K_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sa_m,  # staged_a stride in M dim
    stride_sa_k,  # staged_a stride in K dim
    stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    K_PER_FLAG: tl.constexpr,
    NUM_FLAG_GROUPS_K: tl.constexpr,
    TOTAL_GATHER_TILES: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    NUM_FETCH_STAGES: tl.constexpr,
    GEMM_TILES_PER_STAGE: tl.constexpr,
    FIRST_STAGE_FETCH_SMS: tl.constexpr,
    TRACE: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACE)

    # Interleaved layout with asymmetric first stage:
    #   [fetch0 (P)] [gemm0 (G)] [fetch1 (F)] [gemm1 (G)] ...
    # P = FIRST_STAGE_FETCH_SMS, F = NUM_FETCH_SMS, G = GEMM_TILES_PER_STAGE
    FIRST_STAGE_SIZE: tl.constexpr = FIRST_STAGE_FETCH_SMS + GEMM_TILES_PER_STAGE
    REST_STAGE_SIZE: tl.constexpr = NUM_FETCH_SMS + GEMM_TILES_PER_STAGE
    M_PER_STAGE: tl.constexpr = (NUM_M_TILES + NUM_FETCH_STAGES - 1) // NUM_FETCH_STAGES

    # Two-phase decode: stage 0 has a different size than subsequent stages
    if pid < FIRST_STAGE_SIZE:
        my_stage = zero
        local_pid = pid
        fetch_threshold = zero + FIRST_STAGE_FETCH_SMS
    else:
        adjusted = pid - FIRST_STAGE_SIZE
        my_stage = 1 + adjusted // REST_STAGE_SIZE
        local_pid = adjusted % REST_STAGE_SIZE
        fetch_threshold = zero + NUM_FETCH_SMS

    if local_pid < fetch_threshold:
        # ==============================================================
        # FETCHER  — stage 0 uses FIRST_STAGE_FETCH_SMS WGs,
        #            later stages use NUM_FETCH_SMS WGs
        # ==============================================================
        stage_pid = local_pid

        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().fetch,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=pid,
                pid_n=my_stage,
            )

        src_view = iris.make_tensor_view(A_sharded, M, K_local, stride_am, stride_ak)

        tiles_per_m_group = NUM_FLAG_GROUPS_K * GROUP_SIZE_M

        for const_stage in range(NUM_FETCH_STAGES):
            if my_stage == const_stage:
                stage_fetch_sms = FIRST_STAGE_FETCH_SMS if const_stage == 0 else NUM_FETCH_SMS
                stage_m_start = const_stage * M_PER_STAGE
                stage_m_count = min(M_PER_STAGE, NUM_M_TILES - stage_m_start)
                total_fg_stage = NUM_FLAG_GROUPS_K * stage_m_count

                for fg_idx in range(stage_pid, total_fg_stage, stage_fetch_sms):
                    m_group = fg_idx // tiles_per_m_group
                    within_group = fg_idx % tiles_per_m_group
                    k_flag_group = within_group // GROUP_SIZE_M
                    m_in_group = within_group % GROUP_SIZE_M
                    m_tile = stage_m_start + m_group * GROUP_SIZE_M + m_in_group
                    m_tile = min(m_tile, NUM_M_TILES - 1)
                    k_block_start = k_flag_group * K_PER_FLAG

                    rm = m_tile * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)

                    for k_off in range(K_PER_FLAG):
                        k_block_global = k_block_start + k_off

                        src_rank_idx = k_block_global // NUM_K_BLOCKS_LOCAL
                        k_block_local = k_block_global % NUM_K_BLOCKS_LOCAL

                        pid_m_t = zero + m_tile
                        tile_k_t = zero + k_block_local
                        k_tile = iris.TileView(pid_m_t, tile_k_t, BLOCK_SIZE_M, BLOCK_SIZE_K)

                        rk = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
                        staged_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k

                        for compile_rank in range(world_size):
                            if src_rank_idx == compile_rank:
                                a_tile = ctx.gather(k_tile, src_view, compile_rank, hint=(1, BLOCK_SIZE_K))
                                tl.store(staged_ptrs, a_tile, cache_modifier=".cg")

                    flag_idx = m_tile * NUM_FLAG_GROUPS_K + k_flag_group
                    tl.debug_barrier()  # ensure all per-block stores are visible before setting the flag
                    tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)

    else:
        # ==============================================================
        # GEMM  — gemm_local_id indexes into this stage's M-tile range
        # ==============================================================
        gemm_local_id = local_pid - fetch_threshold
        stage_m_start = my_stage * M_PER_STAGE

        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = gemm_local_id // num_pid_in_group
        first_pid_m = stage_m_start + group_id * GROUP_SIZE_M
        first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((gemm_local_id % num_pid_in_group) % group_sz)
        pid_n = (gemm_local_id % num_pid_in_group) // group_sz
        pid_m = min(pid_m, NUM_M_TILES - 1)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().compute,
                target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1),
                pid_m=pid,
                pid_n=my_stage,
            )

        for k_fg in range(NUM_FLAG_GROUPS_K):
            if TRACE:
                _wait_handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().wait,
                    target_rank=cur_rank,
                    address=flags_ptr + tl.arange(0, 1),
                    pid_m=pid,
                    pid_n=k_fg,
                )

            flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
            while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                pass

            if TRACE:
                ctx.tracing.record_event_end(_wait_handle)

            k_block_base = k_fg * K_PER_FLAG
            for k_off in range(K_PER_FLAG):
                k_block = k_block_base + k_off
                rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                a = tl.load(a_ptrs)

                B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptrs)

                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

        if BIAS:
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_val[:, None]

        c = acc.to(C.type.element_ty)
        C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptrs, c, mask=c_mask, cache_modifier=".wt")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)


# ==========================================================================
# Python API
# ==========================================================================


def all_gather_matmul_hbm_buffer_preamble(
    ctx,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    k_per_flag: Optional[int] = None,
    staged_a_layout: str = "k_contiguous",
) -> FusedWorkspace:
    """
    Allocate workspace.

    Args:
        staged_a_layout: "k_contiguous" (default, row-major (M,K)) or
                         "m_contiguous" (col-major, stored as (K,M) transposed).
    """
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = ctx.get_num_ranks()

    if config is None:
        auto_cfg, auto_kpf, _, _, _ = _auto_config(M, N, K, world_size)
        config = auto_cfg
        if k_per_flag is None:
            k_per_flag = auto_kpf
    if k_per_flag is None:
        k_per_flag = 8  # Safety default; see K-021 best_configs.json for peak perf

    assert world_size * K_local == K
    assert K_local % config.block_size_k == 0
    assert K % config.block_size_k == 0
    assert M % config.block_size_m == 0

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % k_per_flag == 0
    num_flag_groups_k = num_k_blocks // k_per_flag

    ws = FusedWorkspace(
        operation="all_gather_matmul_hbm_buffer",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant=f"hbm_buffer_{staged_a_layout}",
        prepared=True,
    )

    if staged_a_layout == "m_contiguous":
        # Allocate (K, M) row-major, .T gives (M, K) with stride_m=1, stride_k=M
        storage = ctx.zeros((K, M), dtype=A_sharded.dtype)
        ws.aux_buffer = storage.T  # (M, K) view, M-contiguous
    else:
        # Default: (M, K) row-major, stride_m=K, stride_k=1
        ws.aux_buffer = ctx.zeros((M, K), dtype=A_sharded.dtype)

    ws.locks = ctx.zeros((num_m_tiles * num_flag_groups_k,), dtype=torch.int32)

    buffer_mb = M * K * A_sharded.element_size() / (1024**2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    ctx.info(
        f"HBM buffer: staged_a=({M},{K}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"flags={num_m_tiles}x{num_flag_groups_k}, k_per_flag={k_per_flag}"
    )

    ctx.barrier()
    return ws


_EID_FETCH = 1024  # TraceEvent().fetch
_EID_COMPUTE = 2048  # TraceEvent().compute
_EID_WAIT = 3072  # TraceEvent().wait


def _extract_wg_trace(ctx, grid_size, **metadata):
    """Reconstruct per-workgroup trace arrays from DeviceTracing events."""
    import numpy as np

    bufs = ctx.tracing.trace_buffers
    n = min(ctx.tracing.trace_counter.item(), ctx.tracing.max_events)

    event_ids = bufs["event_id"][:n].cpu().numpy()
    pids = bufs["pid"][:n].cpu().numpy()
    timestamps = bufs["timestamp"][:n].cpu().numpy().astype(np.int64)
    # Note: despite the field name, "duration_cycles" stores the absolute end timestamp
    # (set by record_event_end). The actual duration is end_ts - start_ts.
    end_timestamps = bufs["duration_cycles"][:n].cpu().numpy().astype(np.int64)
    xcc_ids = bufs["xcc_id"][:n].cpu().numpy().astype(np.int32)

    starts = torch.zeros(grid_size, dtype=torch.int64)
    ends = torch.zeros(grid_size, dtype=torch.int64)
    waits = torch.zeros(grid_size, dtype=torch.int64)
    xcds = torch.zeros(grid_size, dtype=torch.int32)

    for i in range(n):
        eid = int(event_ids[i])
        wg = int(pids[i])
        if wg >= grid_size:
            continue
        if eid == _EID_FETCH or eid == _EID_COMPUTE:
            starts[wg] = int(timestamps[i])
            ends[wg] = int(end_timestamps[i])
            xcds[wg] = int(xcc_ids[i])
        elif eid == _EID_WAIT:
            waits[wg] += int(end_timestamps[i]) - int(timestamps[i])

    return {"start": starts, "end": ends, "wait": waits, "xcd": xcds, "grid_size": grid_size, **metadata}


def all_gather_matmul_hbm_buffer(
    ctx,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    num_fetch_sms: Optional[int] = None,
    k_per_flag: Optional[int] = None,
    fetch_block_m: Optional[int] = None,
    fetch_block_k: Optional[int] = None,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = 8,
    num_stages: Optional[int] = 2,
    num_fetch_stages: Optional[int] = None,
    first_stage_fetch_sms: Optional[int] = None,
    trace: bool = False,
) -> FusedWorkspace:
    """
    All-gather + matmul with dedicated fetcher/GEMM workgroups.

    When ``config`` is None, uses ``_auto_config()`` to select shape-optimal
    parameters from verified sweep data (K-021). This gives up to 1.44×
    speedup over PyTorch on champion shapes without any manual tuning.

    Args:
        staged_a_layout: Buffer layout for gathered A.
            "k_contiguous" — (M,K) row-major, K is fast dim. Matches NN convention.
            "m_contiguous" — (M,K) with M as fast dim. Matches TN convention (best for tritonblas).
    """
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = ctx.get_num_ranks()

    if config is None:
        # Shape-adaptive auto-config from K-021 sweep data
        auto_cfg, auto_kpf, auto_fs, auto_nfs, auto_fsf = _auto_config(M, N, K, world_size)
        config = auto_cfg
        if k_per_flag is None:
            k_per_flag = auto_kpf
        if num_fetch_sms is None:
            num_fetch_sms = auto_fs
        if num_fetch_stages is None:
            num_fetch_stages = auto_nfs
        if first_stage_fetch_sms is None:
            first_stage_fetch_sms = auto_fsf

    # Apply defaults for any remaining None values (when config is explicit
    # but some params are left at None).
    # kpf=8 is the safety default: +4.3% vs kpf=16 on g6 (IRIS-0018, 934 trials)
    # and avoids kpf=16 validation failures on 2/8 ranks at M=262144.
    # For peak performance on known shapes, use best_configs.json from K-021.
    if k_per_flag is None:
        k_per_flag = 8
    if num_fetch_sms is None:
        num_fetch_sms = 32
    if num_fetch_stages is None:
        num_fetch_stages = 1
    if first_stage_fetch_sms is None:
        first_stage_fetch_sms = 256

    rank = ctx.get_rank()

    assert world_size * K_local == K
    assert output_tensor.shape == (M, N)
    assert M % config.block_size_m == 0
    assert K % config.block_size_k == 0
    assert K_local % config.block_size_k == 0

    if fetch_block_m is None:
        fetch_block_m = config.block_size_m
    if fetch_block_k is None:
        fetch_block_k = config.block_size_k

    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % k_per_flag == 0

    if workspace is None:
        workspace = all_gather_matmul_hbm_buffer_preamble(ctx, A_sharded, B, config, k_per_flag, staged_a_layout)

    workspace.locks.zero_()

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()
    stride_sa_m, stride_sa_k = workspace.aux_buffer.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A_sharded.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_gemm_tiles = num_m_tiles * num_tiles_n
    num_k_blocks_local = K_local // config.block_size_k
    num_flag_groups_k = num_k_blocks // k_per_flag
    total_gather_tiles = num_m_tiles * num_k_blocks

    if num_fetch_sms is None:
        num_fetch_sms = max(1, num_sms // 10)
    assert 0 < num_fetch_sms
    assert num_fetch_stages >= 1

    # First stage can use more fetcher WGs to fill the first GPU wave
    if first_stage_fetch_sms is None:
        first_stage_fetch_sms = num_fetch_sms

    # Interleaved layout: [fetch0 (P)] [gemm0 (G)] [fetch1 (F)] [gemm1 (G)] ...
    m_per_stage = (num_m_tiles + num_fetch_stages - 1) // num_fetch_stages
    gemm_tiles_per_stage = m_per_stage * num_tiles_n
    first_stage_size = first_stage_fetch_sms + gemm_tiles_per_stage
    rest_stage_size = num_fetch_sms + gemm_tiles_per_stage
    total_fetch_wgs = first_stage_fetch_sms + num_fetch_sms * max(0, num_fetch_stages - 1)
    grid_size = first_stage_size + rest_stage_size * max(0, num_fetch_stages - 1)

    if trace:
        max_trace_events = grid_size * 4
        if not ctx.tracing.enabled:
            ctx.tracing.enable(max_events=max_trace_events)
        else:
            ctx.tracing.reset()

    launch_kwargs = {"matrix_instr_nonkdim": 16}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    _hbm_buffer_all_gather_matmul_kernel[(grid_size,)](
        A_sharded,
        B,
        output_tensor,
        bias_ptr,
        workspace.aux_buffer,
        workspace.locks,
        M,
        N,
        K,
        K_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_sa_m,
        stride_sa_k,
        stride_bias,
        ctx.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_fetch_sms,
        num_m_tiles,
        num_tiles_n,
        num_k_blocks,
        num_k_blocks_local,
        k_per_flag,
        num_flag_groups_k,
        total_gather_tiles,
        use_bias,
        config.allow_tf32,
        num_fetch_stages,
        gemm_tiles_per_stage,
        first_stage_fetch_sms,
        trace,
        **launch_kwargs,
    )

    if not async_op:
        ctx.barrier()

    if trace:
        torch.cuda.synchronize()
        workspace.trace_data = _extract_wg_trace(
            ctx,
            grid_size,
            num_fetch_sms=num_fetch_sms,
            num_fetch_stages=num_fetch_stages,
            total_fetch_wgs=total_fetch_wgs,
            num_m_tiles=num_m_tiles,
            num_tiles_n=num_tiles_n,
            first_stage_fetch_sms=first_stage_fetch_sms,
            first_stage_size=first_stage_size,
            rest_stage_size=rest_stage_size,
            gemm_tiles_per_stage=gemm_tiles_per_stage,
        )

    return workspace
