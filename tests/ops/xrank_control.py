
"""Is the failing host-barrier control a harness bug or genuinely stale remote
reads? Rotate the buffer region per iteration so no address is ever re-read.
If that is clean, the data path is fine and stale local L2 on remote lines is
the mechanism. If it still fails, the harness is wrong."""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl

@triton.jit
def produce(buf, n_t, it, base, BLK: tl.constexpr, PWG: tl.constexpr, SCM: tl.constexpr):
    pid = tl.program_id(0); off = tl.arange(0, BLK)
    for t in range(pid, n_t, PWG):
        v = (it * 1000 + t).to(tl.float32) + tl.zeros([BLK], dtype=tl.float32)
        tl.store(buf + base + t * BLK + off, v, cache_modifier=SCM)

@triton.jit
def consume(buf, out, hb: tl.tensor, n_t, base,
            cur: tl.constexpr, W: tl.constexpr, BLK: tl.constexpr,
            CWG: tl.constexpr, LCM: tl.constexpr):
    pid = tl.program_id(0); off = tl.arange(0, BLK)
    for t in range(pid, n_t, CWG):
        s0 = (cur + 1 + pid) % W
        acc = iris.load(buf + base + t * BLK + off, cur, s0, hb, cache_modifier=LCM)
        for i in tl.static_range(1, W):
            acc += iris.load(buf + base + t * BLK + off, cur, (s0 + i) % W, hb, cache_modifier=LCM)
        tl.store(out + t * BLK + off, acc)

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 33); hb = sh.get_heap_bases(); rank = sh.get_rank()
    def P(*a):
        if rank == 0: print(*a, flush=True)
    BLK, n_t, PWG, CWG = 256, 4096, 128, 64
    dev = f"cuda:{rank}"
    t_idx = torch.arange(n_t, device=dev, dtype=torch.float32)
    NIT = 8
    P(f"\nhost-barrier control, ws={W} tiles={n_t}")
    P(f"{'variant':>40} {'iters ok':>9} {'firstBad':>9} {'maxErr':>10}")
    for rotate in [True, False]:
        for lcm in [".cv", None]:
            span = n_t * BLK
            buf = sh.zeros(((NIT + 1) * span if rotate else span,), dtype=torch.float32)
            out = torch.zeros(span, device=dev, dtype=torch.float32)
            sh.barrier()
            ok, fb, me = 0, -1, 0.0
            for it in range(1, NIT + 1):
                base = (it * span) if rotate else 0
                produce[(PWG,)](buf, n_t, it, base, BLK, PWG, ".wt", num_warps=8)
                torch.cuda.synchronize(); sh.barrier()
                consume[(CWG,)](buf, out, hb, n_t, base, rank, W, BLK, CWG, lcm, num_warps=8)
                torch.cuda.synchronize()
                exp = (W * (it * 1000 + t_idx)).repeat_interleave(BLK)
                err = (out - exp).abs().max().item()
                if err > 0.5:
                    if fb < 0: fb = it
                    me = max(me, err)
                else:
                    ok += 1
                sh.barrier()
            P(f"{('rotate' if rotate else 'reuse') + ' buf, load=' + str(lcm):>40} "
              f"{ok:>9} {fb:>9} {me:>10.1f}  {'CLEAN' if fb < 0 else ''}")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
