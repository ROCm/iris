"""FUSED one-shot GEMM+AllReduce in a single kernel (WG specialization).

GEMM WGs: compute this rank's partial into symmetric C, signal per-tile counter
          on EVERY rank (fire-and-forget sys-scope).
Comm WGs: poll LOCAL counter for `ws` arrivals (device-scope), then one-shot
          pull-reduce that tile from all peers into the output.

Producer->consumer, disjoint co-resident WG pools -- the shape that worked for
XCD GEMM+RS, unlike two-shot AR whose AG<->RS dependency is all-to-all.
"""
import os, torch, torch.distributed as dist, triton, triton.language as tl
import iris

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
ws, rank = dist.get_world_size(), dist.get_rank()
DT = torch.float16

def P(*a):
    if rank == 0: print(*a, flush=True)


@triton.jit
def fused_gemm_ar(
    A, B, C_sym, out, flags, heap_bases: tl.tensor,
    M, N, K_local,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n, so_m, so_n,
    flag_target,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GEMM_WGS: tl.constexpr, TOTAL_WGS: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    n_tiles = n_m * n_n

    if pid < GEMM_WGS:
        # ---- producer: GEMM this rank's partial, then signal all ranks ----
        for t in range(pid, n_tiles, GEMM_WGS):
            pm = t // n_n
            pn = t % n_n
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, K_local, BK):
                a = tl.load(A + rm[:, None] * sa_m + (k0 + rk)[None, :] * sa_k,
                            mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K_local), other=0.0)
                b = tl.load(B + (k0 + rk)[:, None] * sb_k + rn[None, :] * sb_n,
                            mask=((k0 + rk)[:, None] < K_local) & (rn[None, :] < N), other=0.0)
                acc += tl.dot(a, b)
            tl.store(C_sym + rm[:, None] * sc_m + rn[None, :] * sc_n,
                     acc.to(C_sym.dtype.element_ty),
                     mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt")
            # signal every rank that our partial for tile t is ready
            tl.atomic_add(flags + t, 1, sem="release", scope="gpu")
            for p in tl.static_range(0, world_size):
                if p != cur_rank:
                    iris.atomic_add(flags + t, 1, cur_rank, p, heap_bases,
                                    sem="release", scope="sys")
    else:
        # ---- consumer: wait for all ranks' partials, one-shot pull-reduce ----
        cid = pid - GEMM_WGS
        comm_wgs = TOTAL_WGS - GEMM_WGS
        for t in range(cid, n_tiles, comm_wgs):
            spins = 0
            done = tl.atomic_add(flags + t, 0, sem="acquire", scope="gpu")
            while (done < flag_target) and (spins < SPIN_LIMIT):
                done = tl.atomic_add(flags + t, 0, sem="acquire", scope="gpu")
                spins += 1
            pm = t // n_n
            pn = t % n_n
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            off = rm[:, None] * sc_m + rn[None, :] * sc_n
            sr = (cur_rank + 1) % world_size
            acc = iris.load(C_sym + off, cur_rank, sr, heap_bases, mask=mask).to(tl.float32)
            for i in tl.static_range(1, world_size):
                r = (sr + i) % world_size
                acc += iris.load(C_sym + off, cur_rank, r, heap_bases, mask=mask).to(tl.float32)
            tl.store(out + rm[:, None] * so_m + rn[None, :] * so_n,
                     acc.to(out.dtype.element_ty), mask=mask)


def bench(fn, n=300, warm=60):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n


shmem = iris.iris(1 << 33)
hb = shmem.get_heap_bases()
N, K = 2880, 4096
K_local = K // ws
MAXCU = torch.cuda.get_device_properties(rank).multi_processor_count

P(f"FUSED one-shot GEMM+AR  ws={ws} N={N} K={K} fp16   (CUs={MAXCU})")

for M in [32, 128]:
    A = torch.randn(M, K_local, device=f"cuda:{rank}", dtype=DT)
    B = torch.randn(K_local, N, device=f"cuda:{rank}", dtype=DT)
    C_sym = shmem.zeros((M, N), dtype=DT)
    out = torch.zeros(M, N, device=f"cuda:{rank}", dtype=DT)
    C_rccl = torch.empty(M, N, device=f"cuda:{rank}", dtype=DT)

    torch_ms = bench(lambda: (torch.mm(A, B, out=C_rccl), dist.all_reduce(C_rccl)))

    torch.mm(A, B, out=C_sym)
    ref = C_sym.clone().float(); dist.all_reduce(ref)

    P("")
    P(f"=== M={M}   torch(mm+AR)={torch_ms:.4f}ms ===")
    P(f"{'BM':>4} {'BN':>5} {'gemmWG':>7} {'commWG':>7} {'ms':>9} {'vs torch':>9} {'diff':>9}")

    results = []
    for BM in [16, 32]:
        if M % BM != 0 and BM > M: continue
        for BN in [64, 128]:
            n_tiles = ((M + BM - 1)//BM) * ((N + BN - 1)//BN)
            for gemm_wgs in [64, 128, 196, 240]:
                for comm_wgs in [32, 64]:
                    total = gemm_wgs + comm_wgs
                    if total > MAXCU: continue
                    flags = shmem.zeros((n_tiles,), dtype=torch.int32)
                    it = [0]
                    def run(BM=BM, BN=BN, gw=gemm_wgs, tw=total, nt=n_tiles):
                        it[0] += 1
                        fused_gemm_ar[(tw,)](
                            A, B, C_sym, out, flags, hb, M, N, K_local,
                            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                            C_sym.stride(0), C_sym.stride(1), out.stride(0), out.stride(1),
                            it[0] * ws, rank, ws, BM, BN, 64, gw, tw, 100000)
                    try:
                        # correctness on a fresh flag buffer
                        flags.zero_(); it[0] = 0
                        shmem.barrier()
                        run()
                        torch.cuda.synchronize()
                        d = (out.float() - ref).abs().max().item()
                        if d >= 1.0:
                            P(f"{BM:>4} {BN:>5} {gemm_wgs:>7} {comm_wgs:>7} {'--':>9} {'--':>9} {d:>9.3f} FAIL")
                            continue
                        ms = bench(run, n=200, warm=40)
                        results.append((ms, BM, BN, gemm_wgs, comm_wgs, d))
                        P(f"{BM:>4} {BN:>5} {gemm_wgs:>7} {comm_wgs:>7} {ms:>9.4f} {torch_ms/ms:>8.2f}x {d:>9.4f}")
                    except Exception as ex:
                        P(f"{BM:>4} {BN:>5} {gemm_wgs:>7} {comm_wgs:>7}  ERR {type(ex).__name__}: {str(ex)[:40]}")

    if results:
        results.sort()
        ms, BM, BN, gw, cw, d = results[0]
        P(f"  BEST FUSED: BM={BM} BN={BN} gemmWG={gw} commWG={cw} -> {ms:.4f}ms ({torch_ms/ms:.2f}x vs torch)")
    else:
        P("  no working fused config")

shmem.barrier()
dist.destroy_process_group()
