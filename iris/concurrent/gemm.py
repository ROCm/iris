# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
``iris.concurrent.gemm`` -- run an independent GEMM concurrently with a collective.

Each function computes a local ``C = A @ B`` **and** an independent collective at
the same time on one device, using work-stealing scheduling. The GEMM and the
collective are *not* producer/consumer -- they share only the GPU.

Two overlap models, selected with ``mode``:

* ``mode="fused"``      -- a single persistent kernel with two work-stealing
  queues; ``gemm_wgs`` workgroups start in the GEMM queue and the rest start in
  the comm queue, and each steals from the other once its home queue drains.
* ``mode="concurrent"`` -- two independent persistent work-stealing kernels
  launched on separate streams; ``gemm_wgs`` / ``comm_wgs`` set each grid.

Example::

    import iris
    shmem = iris.iris(1 << 33)
    A = shmem.randn(M, K); B = shmem.randn(N, K).T
    src = shmem.full((Mc, Nc), shmem.get_rank() + 1.0)
    C, gathered = iris.concurrent.gemm.all_gather(shmem, A, B, src, mode="fused")
"""

import torch
import triton

from . import _kernels

__all__ = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all", "broadcast"]


# Small device-local caches so the hot path does not re-allocate counters/streams.
_counter_cache = {}
_stream_cache = {}


def _counters(device):
    dev = torch.device(device)
    key = (dev.type, dev.index)
    if key not in _counter_cache:
        _counter_cache[key] = torch.zeros(2, dtype=torch.int32, device=dev)
    return _counter_cache[key]


def _streams(device):
    dev = torch.device(device)
    key = (dev.type, dev.index)
    if key not in _stream_cache:
        _stream_cache[key] = (torch.cuda.Stream(), torch.cuda.Stream())
    return _stream_cache[key]


def _arch(device):
    """Short device-arch string for the tuning-db cache key."""
    try:
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", None) or props.name
        return str(arch).split(":", 1)[0]
    except Exception:
        return "unknown"


def _maybe_tune(op, shmem, A, B, comm_src, C, comm_dst, *, collective, mode, num_wgs, default_gemm_block, base_kwargs):
    """Benchmark the top-k predicted configs for this shape and return the best
    ``{gemm_wgs, gemm_block}``. ``C``/``comm_dst`` must already be allocated so
    every candidate run reuses the same output buffers. ``base_kwargs`` are the
    non-tuned kwargs forwarded to ``op`` on each benchmark launch (e.g.
    ``comm_block``, group sizes, ``num_stages``, ``num_warps``)."""
    from . import autotune

    M, K = A.shape
    _, N = B.shape
    Mc, Nc = comm_src.shape
    cu = shmem.get_cu_count()
    nwg = num_wgs if num_wgs is not None else cu
    shape = dict(
        gemm_m=int(M),
        gemm_n=int(N),
        gemm_k=int(K),
        comm_m=int(Mc),
        comm_n=int(Nc),
        collective=collective,
        world=shmem.get_num_ranks(),
    )

    def run(cfg):
        op(
            shmem,
            A,
            B,
            comm_src,
            C=C,
            comm_dst=comm_dst,
            mode=mode,
            num_wgs=num_wgs,
            gemm_wgs=cfg["gemm_wgs"],
            gemm_block=cfg["gemm_block"],
            tune=False,
            **base_kwargs,
        )

    return autotune.tune_config(
        run=run,
        shape=shape,
        mode=mode,
        num_wgs=nwg,
        barrier_fn=shmem.barrier,
        default_gemm_block=default_gemm_block,
        default_gemm_wgs=(nwg * 3) // 4,
        arch=_arch(A.device),
    )


def _maybe_tune_ext(op, collective, shmem, A, B, comm_src, C, comm_dst, p):
    """Tune an ext collective (reduce_scatter / all_to_all / broadcast) in place:
    resolves ``gemm_wgs`` / ``gemm_block`` in the params dict ``p`` built by
    :func:`_common_ext`. No-op if ``gemm_wgs`` is already set (only reached when
    the caller has confirmed ``tune=True``)."""
    if p.get("gemm_wgs") is not None:
        return
    cfg = _maybe_tune(
        op,
        shmem,
        A,
        B,
        comm_src,
        C,
        comm_dst,
        collective=collective,
        mode=p["mode"],
        num_wgs=p["num_wgs"],
        default_gemm_block=p["gemm_block"],
        base_kwargs=dict(
            comm_block=p["comm_block"],
            gemm_group_m=p["gemm_group_m"],
            comm_group_m=p["comm_group_m"],
            num_stages=p["num_stages"],
            num_warps=p["num_warps"],
        ),
    )
    p["gemm_wgs"] = cfg["gemm_wgs"]
    p["gemm_block"] = cfg["gemm_block"]


def all_gather(
    shmem,
    A,
    B,
    comm_src,
    *,
    C=None,
    comm_dst=None,
    mode="fused",
    num_wgs=None,
    gemm_wgs=None,
    comm_wgs=None,
    gemm_block=(256, 64, 64),
    gemm_group_m=6,
    comm_block=(256, 64),
    comm_group_m=6,
    num_stages=2,
    num_warps=8,
    tune=False,
):
    """Concurrent independent GEMM + all-gather.

    Computes ``C = A @ B`` (kept local) while all-gathering ``comm_src`` along
    dim-0 into ``comm_dst`` on every rank.

    Args:
        shmem: Iris context (``iris.iris(...)``).
        A: ``(M, K)`` symmetric tensor.
        B: ``(K, N)`` symmetric tensor.
        comm_src: ``(Mc, Nc)`` symmetric tensor; this rank's all-gather block.
        C: optional ``(M, N)`` output; allocated if ``None``.
        comm_dst: optional ``(world_size*Mc, Nc)`` output; allocated if ``None``.
        mode: ``"fused"`` (one dual-queue kernel) or ``"concurrent"`` (two kernels).
        num_wgs: total persistent workgroups (fused). Default: CU count.
        gemm_wgs: WGs that start in the GEMM queue (fused) / GEMM kernel grid
            (concurrent). Default: fused -> ``num_wgs*3//4``; concurrent ->
            ``cu_count - comm_wgs``.
        comm_wgs: comm kernel grid (concurrent only). Default: ``max(1, cu//8)``.
        gemm_block: ``(BLOCK_M, BLOCK_N, BLOCK_K)`` for the GEMM.
        comm_block: ``(BLOCK_M, BLOCK_N)`` for the all-gather.
        tune: if ``True`` and ``gemm_wgs`` is unset, autotune ``gemm_wgs`` /
            ``gemm_block`` by benchmarking the top-k predicted configs
            (see :mod:`iris.concurrent.autotune`) and cache the winner.

    Returns:
        ``(C, comm_dst)``.
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()
    context_tensor = shmem.get_device_context()

    M, K = A.shape
    _, N = B.shape
    Mc, Nc = comm_src.shape

    if C is None:
        C = shmem.zeros((M, N), device="cuda", dtype=A.dtype)
    if comm_dst is None:
        comm_dst = shmem.zeros((world_size * Mc, Nc), device="cuda", dtype=comm_src.dtype)

    if tune and gemm_wgs is None:
        cfg = _maybe_tune(
            all_gather,
            shmem,
            A,
            B,
            comm_src,
            C,
            comm_dst,
            collective="all_gather",
            mode=mode,
            num_wgs=num_wgs,
            default_gemm_block=gemm_block,
            base_kwargs=dict(
                comm_block=comm_block,
                gemm_group_m=gemm_group_m,
                comm_group_m=comm_group_m,
                num_stages=num_stages,
                num_warps=num_warps,
            ),
        )
        gemm_wgs = cfg["gemm_wgs"]
        gemm_block = cfg["gemm_block"]

    gm, gn, gk = gemm_block
    cm, cn = comm_block
    gemm_total_tiles = triton.cdiv(M, gm) * triton.cdiv(N, gn)
    comm_total_tiles = triton.cdiv(Mc, cm) * triton.cdiv(Nc, cn)
    even_k = (K % gk) == 0

    counters = _counters(A.device)
    gemm_counter = counters[0:1]
    comm_counter = counters[1:2]

    if mode == "fused":
        _num_wgs = num_wgs if num_wgs is not None else cu_count
        _gemm_wgs = gemm_wgs if gemm_wgs is not None else (_num_wgs * 3) // 4
        assert 0 <= _gemm_wgs <= _num_wgs, f"gemm_wgs ({_gemm_wgs}) must be in [0, {_num_wgs}]"

        counters.zero_()
        _kernels.fused_ws_gemm_all_gather[(_num_wgs,)](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            comm_src,
            comm_dst,
            Mc,
            Nc,
            comm_src.stride(0),
            comm_src.stride(1),
            comm_dst.stride(0),
            comm_dst.stride(1),
            gemm_counter,
            comm_counter,
            GEMM_TOTAL_TILES=gemm_total_tiles,
            COMM_TOTAL_TILES=comm_total_tiles,
            GEMM_WGS=_gemm_wgs,
            NUM_WGS=_num_wgs,
            GEMM_BLOCK_M=gm,
            GEMM_BLOCK_N=gn,
            GEMM_BLOCK_K=gk,
            GEMM_GROUP_M=gemm_group_m,
            COMM_BLOCK_M=cm,
            COMM_BLOCK_N=cn,
            COMM_GROUP_M=comm_group_m,
            EVEN_K=even_k,
            context_tensor=context_tensor,
            cur_rank=rank,
            world_size=world_size,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    elif mode == "concurrent":
        _comm_wgs = comm_wgs if comm_wgs is not None else max(1, cu_count // 8)
        _gemm_wgs = gemm_wgs if gemm_wgs is not None else max(1, cu_count - _comm_wgs)

        gemm_stream, comm_stream = _streams(A.device)
        current = torch.cuda.current_stream()

        # Reset both queues on the current stream, then fan out to the two streams.
        counters.zero_()
        gemm_stream.wait_stream(current)
        comm_stream.wait_stream(current)

        with torch.cuda.stream(gemm_stream):
            _kernels.ws_gemm[(_gemm_wgs,)](
                A,
                B,
                C,
                M,
                N,
                K,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                gemm_counter,
                GEMM_TOTAL_TILES=gemm_total_tiles,
                GEMM_BLOCK_M=gm,
                GEMM_BLOCK_N=gn,
                GEMM_BLOCK_K=gk,
                GEMM_GROUP_M=gemm_group_m,
                EVEN_K=even_k,
                num_stages=num_stages,
                num_warps=num_warps,
            )

        with torch.cuda.stream(comm_stream):
            _kernels.ws_all_gather[(_comm_wgs,)](
                comm_src,
                comm_dst,
                Mc,
                Nc,
                comm_src.stride(0),
                comm_src.stride(1),
                comm_dst.stride(0),
                comm_dst.stride(1),
                comm_counter,
                COMM_TOTAL_TILES=comm_total_tiles,
                COMM_BLOCK_M=cm,
                COMM_BLOCK_N=cn,
                COMM_GROUP_M=comm_group_m,
                context_tensor=context_tensor,
                cur_rank=rank,
                world_size=world_size,
                num_warps=num_warps,
            )

        current.wait_stream(gemm_stream)
        current.wait_stream(comm_stream)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'fused' or 'concurrent'")

    return C, comm_dst


def all_reduce(
    shmem,
    A,
    B,
    comm_src,
    *,
    C=None,
    comm_dst=None,
    mode="fused",
    num_wgs=None,
    gemm_wgs=None,
    comm_wgs=None,
    gemm_block=(256, 256, 64),
    gemm_group_m=8,
    comm_block=(256, 64),
    comm_group_m=8,
    num_stages=2,
    num_warps=8,
    tune=False,
):
    """Concurrent independent GEMM + one-shot all-reduce.

    Computes ``C = A @ B`` (kept local) while all-reducing ``comm_src`` (M, N)
    into ``comm_dst`` (M, N), summed across ranks and replicated on every rank.
    Work is block-partitioned: rank r reduces its share of the output tiles
    (reading all peers) and scatters the result to all ranks. Returns ``(C, comm_dst)``.
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()
    heap_bases = shmem.get_heap_bases()

    M, K = A.shape
    _, N = B.shape
    Mc, Nc = comm_src.shape

    if C is None:
        C = shmem.zeros((M, N), device="cuda", dtype=A.dtype)
    if comm_dst is None:
        comm_dst = shmem.zeros((Mc, Nc), device="cuda", dtype=comm_src.dtype)

    if tune and gemm_wgs is None:
        cfg = _maybe_tune(
            all_reduce,
            shmem,
            A,
            B,
            comm_src,
            C,
            comm_dst,
            collective="all_reduce",
            mode=mode,
            num_wgs=num_wgs,
            default_gemm_block=gemm_block,
            base_kwargs=dict(
                comm_block=comm_block,
                gemm_group_m=gemm_group_m,
                comm_group_m=comm_group_m,
                num_stages=num_stages,
                num_warps=num_warps,
            ),
        )
        gemm_wgs = cfg["gemm_wgs"]
        gemm_block = cfg["gemm_block"]

    gm, gn, gk = gemm_block
    cm, cn = comm_block
    gemm_total_tiles = triton.cdiv(M, gm) * triton.cdiv(N, gn)
    ar_total_tiles = triton.cdiv(Mc, cm) * triton.cdiv(Nc, cn)
    ar_tiles_per_rank = triton.cdiv(ar_total_tiles, world_size)
    comm_total_tiles = ar_tiles_per_rank  # per-rank comm queue length
    even_k = (K % gk) == 0

    counters = _counters(A.device)
    gemm_counter = counters[0:1]
    comm_counter = counters[1:2]

    common = dict(
        GEMM_BLOCK_M=gm,
        GEMM_BLOCK_N=gn,
        GEMM_BLOCK_K=gk,
        GEMM_GROUP_M=gemm_group_m,
        COMM_BLOCK_M=cm,
        COMM_BLOCK_N=cn,
        COMM_GROUP_M=comm_group_m,
        EVEN_K=even_k,
        heap_bases=heap_bases,
        cur_rank=rank,
        world_size=world_size,
    )

    if mode == "fused":
        _num_wgs = num_wgs if num_wgs is not None else cu_count
        _gemm_wgs = gemm_wgs if gemm_wgs is not None else (_num_wgs * 3) // 4
        counters.zero_()
        _kernels.fused_ws_gemm_all_reduce[(_num_wgs,)](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            comm_src,
            comm_dst,
            Mc,
            Nc,
            comm_src.stride(0),
            comm_src.stride(1),
            comm_dst.stride(0),
            comm_dst.stride(1),
            gemm_counter,
            comm_counter,
            gemm_total_tiles,
            comm_total_tiles,
            ar_total_tiles,
            ar_tiles_per_rank,
            _gemm_wgs,
            _num_wgs,
            num_stages=num_stages,
            num_warps=num_warps,
            **common,
        )
    elif mode == "concurrent":
        _comm_wgs = comm_wgs if comm_wgs is not None else max(1, cu_count // 8)
        _gemm_wgs = gemm_wgs if gemm_wgs is not None else max(1, cu_count - _comm_wgs)
        gemm_stream, comm_stream = _streams(A.device)
        current = torch.cuda.current_stream()
        counters.zero_()
        gemm_stream.wait_stream(current)
        comm_stream.wait_stream(current)
        with torch.cuda.stream(gemm_stream):
            _kernels.ws_gemm[(_gemm_wgs,)](
                A,
                B,
                C,
                M,
                N,
                K,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                gemm_counter,
                gemm_total_tiles,
                GEMM_BLOCK_M=gm,
                GEMM_BLOCK_N=gn,
                GEMM_BLOCK_K=gk,
                GEMM_GROUP_M=gemm_group_m,
                EVEN_K=even_k,
                num_stages=num_stages,
                num_warps=num_warps,
            )
        with torch.cuda.stream(comm_stream):
            _kernels.ws_all_reduce[(_comm_wgs,)](
                comm_src,
                comm_dst,
                Mc,
                Nc,
                comm_src.stride(0),
                comm_src.stride(1),
                comm_dst.stride(0),
                comm_dst.stride(1),
                comm_counter,
                comm_total_tiles,
                ar_total_tiles,
                ar_tiles_per_rank,
                COMM_BLOCK_M=cm,
                COMM_BLOCK_N=cn,
                COMM_GROUP_M=comm_group_m,
                heap_bases=heap_bases,
                cur_rank=rank,
                world_size=world_size,
                num_warps=num_warps,
            )
        current.wait_stream(gemm_stream)
        current.wait_stream(comm_stream)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'fused' or 'concurrent'")

    return C, comm_dst


# ---------------------------------------------------------------------------
# reduce_scatter / all_to_all / broadcast (generic ext kernels)
# ---------------------------------------------------------------------------
_RS, _A2A, _BC = 2, 3, 4


def _run_ext(
    shmem,
    A,
    B,
    comm_src,
    C,
    comm_dst,
    kind,
    e0,
    comm_total_tiles,
    root,
    mode,
    num_wgs,
    gemm_wgs,
    comm_wgs,
    gemm_block,
    gemm_group_m,
    comm_block,
    comm_group_m,
    num_stages,
    num_warps,
):
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()
    heap_bases = shmem.get_heap_bases()
    M, K = A.shape
    _, N = B.shape
    Mc, Nc = comm_src.shape
    gm, gn, gk = gemm_block
    cm, cn = comm_block
    gemm_total_tiles = triton.cdiv(M, gm) * triton.cdiv(N, gn)
    even_k = (K % gk) == 0
    counters = _counters(A.device)
    gemm_counter, comm_counter = counters[0:1], counters[1:2]
    cst = dict(
        COMM_KIND=kind,
        GEMM_BLOCK_M=gm,
        GEMM_BLOCK_N=gn,
        GEMM_BLOCK_K=gk,
        GEMM_GROUP_M=gemm_group_m,
        COMM_BLOCK_M=cm,
        COMM_BLOCK_N=cn,
        COMM_GROUP_M=comm_group_m,
        EVEN_K=even_k,
        heap_bases=heap_bases,
        cur_rank=rank,
        world_size=world_size,
    )

    if mode == "fused":
        _num_wgs = num_wgs if num_wgs is not None else cu_count
        _gemm_wgs = gemm_wgs if gemm_wgs is not None else (_num_wgs * 3) // 4
        counters.zero_()
        _kernels.fused_ws_gemm_comm_ext[(_num_wgs,)](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            comm_src,
            comm_dst,
            Mc,
            Nc,
            comm_src.stride(0),
            comm_src.stride(1),
            comm_dst.stride(0),
            comm_dst.stride(1),
            e0,
            gemm_counter,
            comm_counter,
            gemm_total_tiles,
            comm_total_tiles,
            _gemm_wgs,
            _num_wgs,
            root,
            num_stages=num_stages,
            num_warps=num_warps,
            **cst,
        )
    elif mode == "concurrent":
        _comm_wgs = comm_wgs if comm_wgs is not None else max(1, cu_count // 8)
        _gemm_wgs = gemm_wgs if gemm_wgs is not None else max(1, cu_count - _comm_wgs)
        gemm_stream, comm_stream = _streams(A.device)
        current = torch.cuda.current_stream()
        counters.zero_()
        gemm_stream.wait_stream(current)
        comm_stream.wait_stream(current)
        with torch.cuda.stream(gemm_stream):
            _kernels.ws_gemm[(_gemm_wgs,)](
                A,
                B,
                C,
                M,
                N,
                K,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                gemm_counter,
                gemm_total_tiles,
                GEMM_BLOCK_M=gm,
                GEMM_BLOCK_N=gn,
                GEMM_BLOCK_K=gk,
                GEMM_GROUP_M=gemm_group_m,
                EVEN_K=even_k,
                num_stages=num_stages,
                num_warps=num_warps,
            )
        with torch.cuda.stream(comm_stream):
            _kernels.ws_comm_ext[(_comm_wgs,)](
                comm_src,
                comm_dst,
                Mc,
                Nc,
                comm_src.stride(0),
                comm_src.stride(1),
                comm_dst.stride(0),
                comm_dst.stride(1),
                e0,
                comm_counter,
                comm_total_tiles,
                root,
                COMM_KIND=kind,
                COMM_BLOCK_M=cm,
                COMM_BLOCK_N=cn,
                COMM_GROUP_M=comm_group_m,
                heap_bases=heap_bases,
                cur_rank=rank,
                world_size=world_size,
                num_warps=num_warps,
            )
        current.wait_stream(gemm_stream)
        current.wait_stream(comm_stream)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'fused' or 'concurrent'")
    return C, comm_dst


def _common_ext(shmem, A, B, C, kw):
    M, K = A.shape
    _, N = B.shape
    if C is None:
        C = shmem.zeros((M, N), device="cuda", dtype=A.dtype)
    return C, dict(
        mode=kw.get("mode", "fused"),
        num_wgs=kw.get("num_wgs"),
        gemm_wgs=kw.get("gemm_wgs"),
        comm_wgs=kw.get("comm_wgs"),
        gemm_block=kw.get("gemm_block", (256, 256, 64)),
        gemm_group_m=kw.get("gemm_group_m", 8),
        comm_block=kw.get("comm_block", (256, 64)),
        comm_group_m=kw.get("comm_group_m", 8),
        num_stages=kw.get("num_stages", 2),
        num_warps=kw.get("num_warps", 8),
    )


def reduce_scatter(shmem, A, B, comm_src, *, C=None, comm_dst=None, **kw):
    """Concurrent GEMM + reduce-scatter: comm_dst (Mc//world, Nc) = this rank's
    slice of the sum of comm_src (Mc, Nc) across ranks. Returns (C, comm_dst)."""
    world_size = shmem.get_num_ranks()
    Mc, Nc = comm_src.shape
    Mout = Mc // world_size
    if comm_dst is None:
        comm_dst = shmem.zeros((Mout, Nc), device="cuda", dtype=comm_src.dtype)
    C, p = _common_ext(shmem, A, B, C, kw)
    if kw.get("tune"):
        _maybe_tune_ext(reduce_scatter, "reduce_scatter", shmem, A, B, comm_src, C, comm_dst, p)
    cm, cn = p["comm_block"]
    comm_total = triton.cdiv(Mout, cm) * triton.cdiv(Nc, cn)
    return _run_ext(shmem, A, B, comm_src, C, comm_dst, _RS, Mout, comm_total, 0, **p)


def all_to_all(shmem, A, B, comm_src, *, C=None, comm_dst=None, **kw):
    """Concurrent GEMM + all-to-all: chunk p of comm_src (Mc, Nc) goes to chunk
    cur_rank of peer p's comm_dst (Mc, Nc). Returns (C, comm_dst)."""
    world_size = shmem.get_num_ranks()
    Mc, Nc = comm_src.shape
    if comm_dst is None:
        comm_dst = shmem.zeros((Mc, Nc), device="cuda", dtype=comm_src.dtype)
    C, p = _common_ext(shmem, A, B, C, kw)
    if kw.get("tune"):
        _maybe_tune_ext(all_to_all, "all_to_all", shmem, A, B, comm_src, C, comm_dst, p)
    cm, cn = p["comm_block"]
    comm_total = triton.cdiv(Mc, cm) * triton.cdiv(Nc, cn)
    return _run_ext(shmem, A, B, comm_src, C, comm_dst, _A2A, Mc // world_size, comm_total, 0, **p)


def broadcast(shmem, A, B, comm_src, *, C=None, comm_dst=None, root=0, **kw):
    """Concurrent GEMM + broadcast of root's comm_src (Mc, Nc) to every rank's
    comm_dst (Mc, Nc). Returns (C, comm_dst)."""
    Mc, Nc = comm_src.shape
    if comm_dst is None:
        comm_dst = shmem.zeros((Mc, Nc), device="cuda", dtype=comm_src.dtype)
    C, p = _common_ext(shmem, A, B, C, kw)
    if kw.get("tune"):
        _maybe_tune_ext(broadcast, "broadcast", shmem, A, B, comm_src, C, comm_dst, p)
    cm, cn = p["comm_block"]
    comm_total = triton.cdiv(Mc, cm) * triton.cdiv(Nc, cn)
    return _run_ext(shmem, A, B, comm_src, C, comm_dst, _BC, 0, comm_total, root, **p)
