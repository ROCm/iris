
"""End-to-end GEMM+AllReduce: torch.mm + RCCL vs torch.mm + best iris collective.
Measured, not summed. Correctness gated against torch."""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, socket
import importlib.util, sys
spec = importlib.util.spec_from_file_location("dt", "tests/ops/dispatch_table.py")
dt_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(dt_mod)
one_k, rs_k, ag_k = dt_mod.one_k, dt_mod.rs_k, dt_mod.ag_k
NG, KG = 2880, 4096

def bench(fn, n=100, w=25):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(w): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n

# best configs from the gated dispatch sweep
BEST = {128:  ("one", 16, 64, 196, None, None, None, None, None, None),
        256:  ("two", 32, 64, 256, 64, 128, 256, None, None, None),
        512:  ("two", 16, 64, 196, 32, 256, 256, None, None, None),
        1024: ("two", 16, 64, 196, 16, 128, 256, None, None, None),
        2048: ("two", 16, 128, 196, 16, 128, 256, None, None, None)}

def _w(lr, W, url):
    dist.init_process_group(backend="nccl", init_method=url, world_size=W, rank=lr)
    torch.cuda.set_device(lr)
    sh = iris.iris(1 << 33); hb = sh.get_heap_bases(); rank = sh.get_rank(); dt = torch.float16
    KL = KG // W
    def P(*a):
        if rank == 0: print(*a, flush=True)
    P(f"\nE2E GEMM+AllReduce measured, ws={W} N={NG} K={KG} fp16")
    P(f"{'M':>6} {'torch mm+AR':>12} {'iris mm+coll':>13} {'speedup':>8} {'maxdiff':>9} {'pattern':>8}")
    for M in [128, 256, 512, 1024, 2048]:
        MS = M // W
        A = sh.zeros((M, KL), device="cuda", dtype=dt)
        A.copy_(torch.randn(M, KL, dtype=dt, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(KL, NG, dtype=dt, device=f"cuda:{rank}") * 0.1
        C = sh.zeros((M, NG), device="cuda", dtype=dt)
        out = torch.zeros(M, NG, device=f"cuda:{rank}", dtype=dt)
        shard = sh.zeros((MS, NG), device="cuda", dtype=dt)
        ref = torch.mm(A, B); dist.all_reduce(ref); torch.cuda.synchronize()
        tmp = torch.zeros_like(out)
        t_ms = bench(lambda: (torch.mm(A, B, out=tmp), dist.all_reduce(tmp)))
        pat = BEST[M][0]
        if pat == "one":
            _, bm, bn, wg = BEST[M][:4]
            def run():
                torch.mm(A, B, out=C)
                one_k[(wg,)](C, out, hb, M, NG, C.stride(0), C.stride(1),
                             out.stride(0), out.stride(1), rank, W, bm, bn, wg, num_warps=8)
        else:
            _, rbm, rbn, rwg, abm, abn, awg = BEST[M][:7]
            def run():
                torch.mm(A, B, out=C)
                rs_k[(rwg,)](C, shard, hb, M, NG, C.stride(0), C.stride(1),
                             shard.stride(0), shard.stride(1), rank, W, rbm, rbn, rwg, num_warps=8)
                sh.barrier()
                ag_k[(awg,)](shard, out, hb, M, NG, shard.stride(0), shard.stride(1),
                             out.stride(0), out.stride(1), rank, W, abm, abn, awg, num_warps=8)
                out[rank*MS:(rank+1)*MS].copy_(shard)
        sh.barrier(); out.zero_(); run(); torch.cuda.synchronize()
        d = (out.float() - ref.float()).abs().max().item()
        sh.barrier()
        if d > 0.05:
            P(f"{M:>6} {t_ms:>12.4f} {'--':>13} {'--':>8} {d:>9.4f} {pat:>8} FAIL"); continue
        ms = bench(run)
        P(f"{M:>6} {t_ms:>12.4f} {ms:>13.4f} {t_ms/ms:>7.2f}x {d:>9.4f} {pat:>8}")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_w, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"), nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
