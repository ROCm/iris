
"""Static grid-stride vs dynamic atomic counter when workers arrive staggered.

Models the elastic-pool idea: AWG workers start the AllGather immediately, the
rest burn a GEMM-shaped delay first and only then join. STATIC partitions tiles
across all workers up front, so a tile owned by a late worker waits for it.
DYNAMIC hands out tiles from an atomic counter, so early workers absorb the
backlog and late arrivals take whatever is left.
"""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl
NG = 2880

def bench(fn, n=50, w=15):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(w): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n

@triton.jit
def ag_elastic(S, o, ctr, sink, hb: tl.tensor, M, N, ssm, ssn, som, son, base,
               cur: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               AWG: tl.constexpr, TWG: tl.constexpr, DELAY: tl.constexpr,
               DYNAMIC: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    MS = M // W; n_n = tl.cdiv(N, BN); n_pt = tl.cdiv(MS, BM) * n_n
    n_t = n_pt * (W - 1)
    # late workers pay a GEMM-shaped delay before joining
    if pid >= AWG:
        acc = tl.zeros([64], dtype=tl.float32) + pid.to(tl.float32)
        for _ in range(DELAY):
            acc = acc * 1.0000001 + 1.0
        tl.store(sink + pid * 64 + tl.arange(0, 64), acc)
    if DYNAMIC:
        b = tl.atomic_add(ctr, CHUNK, sem="relaxed", scope="gpu")
        while b < n_t:
            for k in tl.static_range(0, CHUNK):
                t = b + k
                pk = t % (W - 1); lt = t // (W - 1); src = (cur + 1 + pk) % W
                pm = lt // n_n; pn = lt % n_n
                rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
                mk = (rm[:, None] < MS) & (rn[None, :] < N) & (t < n_t)
                v = iris.load(S + base + rm[:, None] * ssm + rn[None, :] * ssn,
                              cur, src, hb, mask=mk)
                om = src * MS + pm * BM + tl.arange(0, BM)
                tl.store(o + om[:, None] * som + rn[None, :] * son, v, mask=mk)
            b = tl.atomic_add(ctr, CHUNK, sem="relaxed", scope="gpu")
    else:
        for t in range(pid, n_t, TWG):
            pk = t % (W - 1); lt = t // (W - 1); src = (cur + 1 + pk) % W
            pm = lt // n_n; pn = lt % n_n
            rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
            mk = (rm[:, None] < MS) & (rn[None, :] < N)
            v = iris.load(S + base + rm[:, None] * ssm + rn[None, :] * ssn, cur, src, hb, mask=mk)
            om = src * MS + pm * BM + tl.arange(0, BM)
            tl.store(o + om[:, None] * som + rn[None, :] * son, v, mask=mk)


def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 33); hb = sh.get_heap_bases(); rank = sh.get_rank(); dt = torch.float16
    def P(*a):
        if rank == 0: print(*a, flush=True)
    M = 2048; MS = M // W
    S = sh.randn((MS, NG), dtype=dt)
    o = torch.zeros(M, NG, device=f"cuda:{rank}", dtype=dt)
    ctr = sh.zeros((1,), dtype=torch.int32)
    sink = torch.zeros(512 * 64, device=f"cuda:{rank}", dtype=torch.float32)
    gath = [torch.empty_like(S) for _ in range(W)]
    dist.all_gather(gath, S); ref = torch.cat(gath, 0)
    P(f"\nelastic AG dispatch, M={M} ws={W}. AWG = workers present at t=0, "
      f"rest join after DELAY")
    P(f"{'AWG':>5} {'TWG':>5} {'delay':>7} {'static':>10} {'dyn c1':>8} {'dyn c8':>8} "
      f"{'dyn c32':>8} {'best dyn':>9} {'ok':>5}")
    for AWG, TWG in [(32, 224), (64, 224)]:
        for DELAY in [0, 2000, 8000]:
            res = {}
            for dyn, ch in [(0,1),(1,1),(1,8),(1,32)]:
                def run(dyn=dyn, ch=ch, AWG=AWG, TWG=TWG, DELAY=DELAY):
                    if dyn: ctr.zero_()
                    ag_elastic[(TWG,)](S, o, ctr, sink, hb, M, NG, S.stride(0), S.stride(1),
                                       o.stride(0), o.stride(1), 0, rank, W, 16, 128,
                                       AWG, TWG, DELAY, dyn, ch, num_warps=8)
                sh.barrier(); o.zero_(); run(); torch.cuda.synchronize()
                o[rank*MS:(rank+1)*MS].copy_(S)
                d = (o.float() - ref.float()).abs().max().item()
                sh.barrier()
                res[(dyn,ch)] = (bench(run), d)
            st, sd = res[(0,1)]
            d1, _ = res[(1,1)]; d8, _ = res[(1,8)]; d32, dd = res[(1,32)]
            ok = "yes" if max(sd, dd) < 0.05 else "FAIL"
            P(f"{AWG:>5} {TWG:>5} {DELAY:>7} {st:>10.4f} {d1:>8.4f} {d8:>8.4f} {d32:>8.4f} "
              f"{st/min(d1,d8,d32):>8.2f}x {ok:>5}")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
