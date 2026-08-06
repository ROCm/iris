
"""Correctness-gated collective comparison across M: RCCL vs one-shot pull vs
fixed two-shot (peer-staggered RS + peer-interleaved AG). Produces the dispatch table."""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl
NG = 2880

def bench(fn, n=100, w=25):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(w): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n

@triton.jit
def one_k(C, o, hb: tl.tensor, M, N, scm, scn, som, son,
          cur: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, WGS: tl.constexpr):
    pid = tl.program_id(0); n_n = tl.cdiv(N, BN); n_t = tl.cdiv(M, BM) * n_n
    for t in range(pid, n_t, WGS):
        pm = t // n_n; pn = t % n_n
        rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
        mk = (rm[:, None] < M) & (rn[None, :] < N); off = rm[:, None] * scm + rn[None, :] * scn
        s0 = (cur + 1 + pid) % W
        acc = iris.load(C + off, cur, s0, hb, mask=mk).to(tl.float32)
        for i in tl.static_range(1, W):
            acc += iris.load(C + off, cur, (s0 + i) % W, hb, mask=mk).to(tl.float32)
        tl.store(o + rm[:, None] * som + rn[None, :] * son, acc.to(o.dtype.element_ty), mask=mk)

@triton.jit
def rs_k(C, o, hb: tl.tensor, M, N, scm, scn, som, son,
         cur: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, WGS: tl.constexpr):
    pid = tl.program_id(0); MS = M // W; n_n = tl.cdiv(N, BN); n_t = tl.cdiv(MS, BM) * n_n
    r0 = cur * MS
    for t in range(pid, n_t, WGS):
        pm = t // n_n; pn = t % n_n
        rm = r0 + pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
        mk = (rm[:, None] < r0 + MS) & (rn[None, :] < N); off = rm[:, None] * scm + rn[None, :] * scn
        s0 = (cur + 1 + pid) % W
        acc = iris.load(C + off, cur, s0, hb, mask=mk).to(tl.float32)
        for i in tl.static_range(1, W):
            acc += iris.load(C + off, cur, (s0 + i) % W, hb, mask=mk).to(tl.float32)
        om = pm * BM + tl.arange(0, BM)
        tl.store(o + om[:, None] * som + rn[None, :] * son, acc.to(o.dtype.element_ty),
                 mask=(om[:, None] < MS) & (rn[None, :] < N))

@triton.jit
def ag_k(S, o, hb: tl.tensor, M, N, ssm, ssn, som, son,
         cur: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, WGS: tl.constexpr):
    pid = tl.program_id(0); MS = M // W; n_n = tl.cdiv(N, BN); n_pt = tl.cdiv(MS, BM) * n_n
    n_t = n_pt * (W - 1)
    for t in range(pid, n_t, WGS):
        pk = t % (W - 1); lt = t // (W - 1); src = (cur + 1 + pk) % W
        pm = lt // n_n; pn = lt % n_n
        rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
        mk = (rm[:, None] < MS) & (rn[None, :] < N)
        v = iris.load(S + rm[:, None] * ssm + rn[None, :] * ssn, cur, src, hb, mask=mk)
        om = src * MS + pm * BM + tl.arange(0, BM)
        tl.store(o + om[:, None] * som + rn[None, :] * son, v, mask=mk)

CFGS = [(16,64),(32,64),(16,128),(32,128),(64,128),(32,256)]
WGSS = [32,64,128,196,256]

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 33); hb = sh.get_heap_bases(); rank = sh.get_rank(); dt = torch.float16
    def P(*a):
        if rank == 0: print(*a, flush=True)
    P(f"\ncollective-only comparison, ws={W}, N={NG}, fp16, correctness-gated")
    P(f"{'M':>6} {'RCCL ms':>9} {'1shot ms':>9} {'2shot ms':>9} {'RS':>8} {'AG':>8} "
      f"{'best':>8} {'vs RCCL':>8} {'ok':>4}")
    for M in [128, 256, 512, 1024, 2048]:
        MS = M // W
        C = sh.randn((M, NG), dtype=dt)
        S = sh.randn((MS, NG), dtype=dt)
        o1 = torch.zeros(M, NG, device=f"cuda:{rank}", dtype=dt)
        rso = torch.zeros(MS, NG, device=f"cuda:{rank}", dtype=dt)
        ago = torch.zeros(M, NG, device=f"cuda:{rank}", dtype=dt)
        X = C.clone()
        rccl = bench(lambda: dist.all_reduce(X))
        full = C.clone(); dist.all_reduce(full)
        gath = [torch.empty_like(S) for _ in range(W)]
        dist.all_gather(gath, S); ag_ref = torch.cat(gath, 0)
        def sweep(fn, out, args, ref, refslice=None, fill=None):
            best = (9e9, None); bd = None
            for BM, BN in CFGS:
                if BM > (MS if refslice else M): continue
                for wg in WGSS:
                    sh.barrier()
                    out.zero_()
                    fn[(wg,)](*args, rank, W, BM, BN, wg, num_warps=8)
                    torch.cuda.synchronize()
                    if fill is not None: out[rank*MS:(rank+1)*MS].copy_(fill)
                    d = (out.float() - ref.float()).abs().max().item()
                    if d > 0.05: continue
                    ms = bench(lambda BM=BM, BN=BN, wg=wg: fn[(wg,)](*args, rank, W, BM, BN, wg, num_warps=8))
                    if ms < best[0]: best = (ms, f"{BM}x{BN}/{wg}"); bd = d
            return best, bd
        b1, d1 = sweep(one_k, o1, (C, o1, hb, M, NG, C.stride(0), C.stride(1), o1.stride(0), o1.stride(1)), full)
        brs, drs = sweep(rs_k, rso, (C, rso, hb, M, NG, C.stride(0), C.stride(1), rso.stride(0), rso.stride(1)),
                         full[rank*MS:(rank+1)*MS], refslice=True)
        bag, dag = sweep(ag_k, ago, (S, ago, hb, M, NG, S.stride(0), S.stride(1), ago.stride(0), ago.stride(1)),
                         ag_ref, fill=S)
        two = brs[0] + bag[0]
        best = min(b1[0], two)
        ok = "yes" if (d1 is not None and drs is not None and dag is not None) else "NO"
        P(f"{M:>6} {rccl:>9.4f} {b1[0]:>9.4f} {two:>9.4f} {brs[0]:>8.4f} {bag[0]:>8.4f} "
          f"{best:>8.4f} {rccl/best:>7.2f}x {ok:>4}")
        P(f"       cfg  1shot={b1[1]}  RS={brs[1]}  AG={bag[1]}   "
          f"maxdiff 1shot={d1} RS={drs} AG={dag}")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
