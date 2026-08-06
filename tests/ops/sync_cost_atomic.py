
"""Re-test the cost of synchronization using an ATOMIC poll, which cannot be
hoisted out of the spin loop. The earlier no-wait result compared two
volatile-load polls, and volatile does not prevent LICM on the AMD backend, so
both sides may have been running without synchronizing. Fresh flags per config,
correctness gated."""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl

@triton.jit
def fused(A, B, C, out, flags, hb: tl.tensor, M, N, KL,
          sam, sak, sbk, sbn, scm, scn, som, son, target,
          cur: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, GWG: tl.constexpr, TWG: tl.constexpr, SPIN: tl.constexpr,
          MODE: tl.constexpr):
    # MODE 0 = atomic poll (correct), 1 = volatile-load poll, 2 = no wait at all
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN); n_t = tl.cdiv(M, BM) * n_n
    if pid < GWG:
        for t in range(pid, n_t, GWG):
            pm = t // n_n; pn = t % n_n
            rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK); mk = (rm[:, None] < M) & (rn[None, :] < N)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, KL, BK):
                a = tl.load(A + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                            mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < KL), other=0.0)
                b = tl.load(B + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                            mask=((k0 + rk)[:, None] < KL) & (rn[None, :] < N), other=0.0)
                acc += tl.dot(a, b)
            tl.store(C + rm[:, None] * scm + rn[None, :] * scn,
                     acc.to(C.dtype.element_ty), mask=mk, cache_modifier=".wt")
            tl.debug_barrier()
            tl.atomic_add(flags + t, 1, sem="release", scope="gpu")
            for p in tl.static_range(0, W):
                if p != cur:
                    iris.atomic_add(flags + t, 1, cur, p, hb, sem="release", scope="sys")
    else:
        cid = pid - GWG; cwg = TWG - GWG
        for t in range(cid, n_t, cwg):
            if MODE == 0:
                d = tl.atomic_add(flags + t, 0, sem="acquire", scope="sys"); s = 0
                while (d < target) and (s < SPIN):
                    d = tl.atomic_add(flags + t, 0, sem="acquire", scope="sys"); s += 1
            elif MODE == 1:
                d = tl.load(flags + t, volatile=True); s = 0
                while (d < target) and (s < SPIN):
                    d = tl.load(flags + t, volatile=True); s += 1
            pm = t // n_n; pn = t % n_n
            rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
            mk = (rm[:, None] < M) & (rn[None, :] < N)
            off = rm[:, None] * scm + rn[None, :] * scn
            s0 = (cur + 1 + cid) % W
            acc = iris.load(C + off, cur, s0, hb, mask=mk).to(tl.float32)
            for i in tl.static_range(1, W):
                acc += iris.load(C + off, cur, (s0 + i) % W, hb, mask=mk).to(tl.float32)
            tl.store(out + rm[:, None] * som + rn[None, :] * son,
                     acc.to(out.dtype.element_ty), mask=mk)

def bench(f, n=50, w=15):
    for _ in range(w): f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 33); hb = sh.get_heap_bases(); rank = sh.get_rank(); dt = torch.float16
    N, K = 2880, 4096; KL = K // W
    def P(*a):
        if rank == 0: print(*a, flush=True)
    for M in [2048]:
        BM, BN = 32, 128
        A = torch.randn(M, KL, device=f"cuda:{rank}", dtype=dt) * 0.1
        B = torch.randn(KL, N, device=f"cuda:{rank}", dtype=dt) * 0.1
        C = sh.zeros((M, N), dtype=dt)
        out = torch.zeros(M, N, device=f"cuda:{rank}", dtype=dt)
        n_t = ((M + BM - 1) // BM) * ((N + BN - 1) // BN)
        torch.mm(A, B, out=C); ref = C.clone().float(); dist.all_reduce(ref)
        P(f"\nM={M} tiles={n_t} ws={W}  -- fresh flags per config, correctness gated")
        P(f"{'poll':>16} {'G/C':>8} {'ms':>9} {'maxdiff':>9} {'verdict':>8}")
        for mode, nm in [(0, "atomic"), (1, "volatile-load"), (2, "NO WAIT")]:
            for g, c in [(128, 64), (192, 32)]:
                flags = sh.zeros((n_t,), dtype=torch.int32)
                it = [0]
                def run(mode=mode, g=g, c=c, flags=flags):
                    it[0] += 1
                    fused[(g + c,)](A, B, C, out, flags, hb, M, N, KL,
                                    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                    C.stride(0), C.stride(1), out.stride(0), out.stride(1),
                                    it[0] * W, rank, W, BM, BN, 64, g, g + c,
                                    2_000_000, mode, num_warps=8)
                sh.barrier(); out.zero_(); run(); torch.cuda.synchronize()
                d = (out.float() - ref).abs().max().item()
                sh.barrier()
                ms = bench(run)
                P(f"{nm:>16} {f'{g}/{c}':>8} {ms:>9.4f} {d:>9.4f} "
                  f"{'OK' if d < 0.5 else 'WRONG':>8}")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
