
"""Does anything fix the cross-rank handoff? Adds a host-barrier control:
if the two-kernel + host-barrier path is clean, the data path is fine and the
problem is purely ordering inside the fused kernel. If it also fails, the
harness is wrong -- check that before theorising."""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl

@triton.jit
def produce(buf, flags, hb: tl.tensor, n_t, it,
            cur: tl.constexpr, W: tl.constexpr, BLK: tl.constexpr,
            PWG: tl.constexpr, SCM: tl.constexpr, FSTRIDE: tl.constexpr,
            FENCE: tl.constexpr, SIGNAL: tl.constexpr):
    pid = tl.program_id(0); off = tl.arange(0, BLK)
    for t in range(pid, n_t, PWG):
        v = (it * 1000 + t).to(tl.float32) + tl.zeros([BLK], dtype=tl.float32)
        tl.store(buf + t * BLK + off, v, cache_modifier=SCM)
        if FENCE:
            tl.debug_barrier()
        if SIGNAL:
            fi = t * FSTRIDE
            tl.atomic_add(flags + fi, 1, sem="release", scope="gpu")
            for p in tl.static_range(0, W):
                if p != cur:
                    iris.atomic_add(flags + fi, 1, cur, p, hb, sem="release", scope="sys")

@triton.jit
def consume(buf, out, flags, hb: tl.tensor, n_t, it,
            cur: tl.constexpr, W: tl.constexpr, BLK: tl.constexpr,
            CWG: tl.constexpr, LCM: tl.constexpr, FSTRIDE: tl.constexpr,
            SPIN: tl.constexpr, WAIT: tl.constexpr):
    pid = tl.program_id(0); off = tl.arange(0, BLK)
    for t in range(pid, n_t, CWG):
        if WAIT:
            fi = t * FSTRIDE; tgt = it * W
            d = tl.load(flags + fi, volatile=True); s = 0
            while (d < tgt) and (s < SPIN):
                d = tl.load(flags + fi, volatile=True); s += 1
            _ = tl.atomic_add(flags + fi, 0, sem="acquire", scope="sys")
        s0 = (cur + 1 + pid) % W
        acc = iris.load(buf + t * BLK + off, cur, s0, hb, cache_modifier=LCM)
        for i in tl.static_range(1, W):
            acc += iris.load(buf + t * BLK + off, cur, (s0 + i) % W, hb, cache_modifier=LCM)
        tl.store(out + t * BLK + off, acc)

@triton.jit
def fused(buf, out, flags, hb: tl.tensor, n_t, it,
          cur: tl.constexpr, W: tl.constexpr, BLK: tl.constexpr,
          PWG: tl.constexpr, TWG: tl.constexpr, SCM: tl.constexpr, LCM: tl.constexpr,
          FSTRIDE: tl.constexpr, SPIN: tl.constexpr, FENCE: tl.constexpr):
    pid = tl.program_id(0); off = tl.arange(0, BLK)
    if pid < PWG:
        for t in range(pid, n_t, PWG):
            v = (it * 1000 + t).to(tl.float32) + tl.zeros([BLK], dtype=tl.float32)
            tl.store(buf + t * BLK + off, v, cache_modifier=SCM)
            if FENCE:
                tl.debug_barrier()
            fi = t * FSTRIDE
            tl.atomic_add(flags + fi, 1, sem="release", scope="gpu")
            for p in tl.static_range(0, W):
                if p != cur:
                    iris.atomic_add(flags + fi, 1, cur, p, hb, sem="release", scope="sys")
    else:
        cid = pid - PWG; cwg = TWG - PWG
        for t in range(cid, n_t, cwg):
            fi = t * FSTRIDE; tgt = it * W
            d = tl.load(flags + fi, volatile=True); s = 0
            while (d < tgt) and (s < SPIN):
                d = tl.load(flags + fi, volatile=True); s += 1
            _ = tl.atomic_add(flags + fi, 0, sem="acquire", scope="sys")
            s0 = (cur + 1 + cid) % W
            acc = iris.load(buf + t * BLK + off, cur, s0, hb, cache_modifier=LCM)
            for i in tl.static_range(1, W):
                acc += iris.load(buf + t * BLK + off, cur, (s0 + i) % W, hb, cache_modifier=LCM)
            tl.store(out + t * BLK + off, acc)

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 32); hb = sh.get_heap_bases(); rank = sh.get_rank()
    def P(*a):
        if rank == 0: print(*a, flush=True)
    BLK, n_t, FSTRIDE = 256, 4096, 32
    PWG, TWG = 128, 192
    dev = f"cuda:{rank}"
    t_idx = torch.arange(n_t, device=dev, dtype=torch.float32)
    P(f"\nhandoff variants  ws={W} tiles={n_t}")
    P(f"{'variant':>34} {'iters ok':>9} {'firstBad':>9} {'maxErr':>10}")
    def run_variant(name, fn):
        buf = sh.zeros((n_t * BLK,), dtype=torch.float32)
        out = torch.zeros(n_t * BLK, device=dev, dtype=torch.float32)
        flags = sh.zeros((n_t * FSTRIDE,), dtype=torch.int32)
        sh.barrier()
        ok, fb, me = 0, -1, 0.0
        for it in range(1, 9):
            fn(buf, out, flags, it)
            torch.cuda.synchronize()
            exp = (W * (it * 1000 + t_idx)).repeat_interleave(BLK)
            err = (out - exp).abs().max().item()
            if err > 0.5:
                if fb < 0: fb = it
                me = max(me, err)
            else:
                ok += 1
        sh.barrier()
        P(f"{name:>34} {ok:>9} {fb:>9} {me:>10.1f}  {'CLEAN' if fb < 0 else ''}")
    # control: two kernels with a HOST barrier between them
    def ctrl(buf, out, flags, it):
        produce[(PWG,)](buf, flags, hb, n_t, it, rank, W, BLK, PWG, ".wt", FSTRIDE,
                        False, False, num_warps=8)
        torch.cuda.synchronize(); sh.barrier()
        consume[(64,)](buf, out, flags, hb, n_t, it, rank, W, BLK, 64, ".cv", FSTRIDE,
                       2_000_000, False, num_warps=8)
    run_variant("CONTROL two-kernel + host barrier", ctrl)
    # same two kernels, flags instead of the host barrier
    def flagged(buf, out, flags, it):
        produce[(PWG,)](buf, flags, hb, n_t, it, rank, W, BLK, PWG, ".wt", FSTRIDE,
                        True, True, num_warps=8)
        consume[(64,)](buf, out, flags, hb, n_t, it, rank, W, BLK, 64, ".cv", FSTRIDE,
                       2_000_000, True, num_warps=8)
    run_variant("two-kernel + flags (no host barrier)", flagged)
    for fence in [True, False]:
        for scm, lcm in [(".wt", ".cv"), (None, ".cv"), (".wt", None)]:
            def f(buf, out, flags, it, scm=scm, lcm=lcm, fence=fence):
                fused[(TWG,)](buf, out, flags, hb, n_t, it, rank, W, BLK, PWG, TWG,
                              scm, lcm, FSTRIDE, 2_000_000, fence, num_warps=8)
            run_variant(f"fused fence={int(fence)} st={scm} ld={lcm}", f)
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
