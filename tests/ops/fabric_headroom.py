
"""Does the fabric have headroom for a second concurrent comm pool?

RS and AG run as two disjoint workgroup pools in ONE kernel with no dependency
between them -- pure bandwidth, no flags. If concurrent time is near max(alone)
the fabric has headroom and a serialized AG really was wasting idle links. If it
is near sum(alone) the fabric is already saturated and fanning AG out can only
reclaim what RS was not using.
"""
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
def combo(C, S, rso, ago, hb: tl.tensor, M, N, scm, scn, ssm, ssn, som, son, aom, aon,
          cur: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          RWG: tl.constexpr, TWG: tl.constexpr, DO_RS: tl.constexpr, DO_AG: tl.constexpr):
    pid = tl.program_id(0); MS = M // W; n_n = tl.cdiv(N, BN)
    if DO_RS and (pid < RWG):
        n_t = tl.cdiv(MS, BM) * n_n; r0 = cur * MS
        for t in range(pid, n_t, RWG):
            pm = t // n_n; pn = t % n_n
            rm = r0 + pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
            mk = (rm[:, None] < r0 + MS) & (rn[None, :] < N)
            off = rm[:, None] * scm + rn[None, :] * scn
            s0 = (cur + 1 + pid) % W
            acc = iris.load(C + off, cur, s0, hb, mask=mk).to(tl.float32)
            for i in tl.static_range(1, W):
                acc += iris.load(C + off, cur, (s0 + i) % W, hb, mask=mk).to(tl.float32)
            om = pm * BM + tl.arange(0, BM)
            tl.store(rso + om[:, None] * som + rn[None, :] * son, acc.to(rso.dtype.element_ty),
                     mask=(om[:, None] < MS) & (rn[None, :] < N))
    if DO_AG and (pid >= RWG):
        cid = pid - RWG; cwg = TWG - RWG
        n_pt = tl.cdiv(MS, BM) * n_n; n_t = n_pt * (W - 1)
        for t in range(cid, n_t, cwg):
            pk = t % (W - 1); lt = t // (W - 1); src = (cur + 1 + pk) % W
            pm = lt // n_n; pn = lt % n_n
            rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
            mk = (rm[:, None] < MS) & (rn[None, :] < N)
            v = iris.load(S + rm[:, None] * ssm + rn[None, :] * ssn, cur, src, hb, mask=mk)
            om = src * MS + pm * BM + tl.arange(0, BM)
            tl.store(ago + om[:, None] * aom + rn[None, :] * aon, v, mask=mk)

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 33); hb = sh.get_heap_bases(); rank = sh.get_rank(); dt = torch.float16
    def P(*a):
        if rank == 0: print(*a, flush=True)
    M = 2048; MS = M // W
    C = sh.randn((M, NG), dtype=dt); S = sh.randn((MS, NG), dtype=dt)
    rso = torch.zeros(MS, NG, device=f"cuda:{rank}", dtype=dt)
    ago = torch.zeros(M, NG, device=f"cuda:{rank}", dtype=dt)
    args = (C, S, rso, ago, hb, M, NG, C.stride(0), C.stride(1), S.stride(0), S.stride(1),
            rso.stride(0), rso.stride(1), ago.stride(0), ago.stride(1))
    rs_b = M * NG * 2; ag_b = (W - 1) * MS * NG * 2
    P(f"\nconcurrent comm pools, M={M} ws={W}. RS reads {rs_b/1e6:.1f}MB, AG {ag_b/1e6:.1f}MB")
    P(f"{'variant':>28} {'RWG':>4} {'AWG':>4} {'ms':>8} {'GB/s':>8} {'%line':>6}")
    def run(rwg, twg, do_rs, do_ag, bm=16, bn=128):
        return bench(lambda: combo[(twg,)](*args, rank, W, bm, bn, rwg, twg, do_rs, do_ag,
                                           num_warps=8))
    for rwg, awg in [(196, 0), (0, 256), (196, 256), (128, 128), (96, 96), (64, 64)]:
        twg = rwg + awg
        do_rs = rwg > 0; do_ag = awg > 0
        if twg == 0: continue
        sh.barrier()
        ms = run(rwg, twg, do_rs, do_ag)
        byts = (rs_b if do_rs else 0) + (ag_b if do_ag else 0)
        nm = ("RS alone" if not do_ag else "AG alone" if not do_rs else "RS+AG concurrent")
        P(f"{nm:>28} {rwg:>4} {awg:>4} {ms:>8.4f} {byts/(ms*1e-3)/1e9:>8.1f} "
          f"{byts/(ms*1e-3)/1e9/448*100:>5.0f}%")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
