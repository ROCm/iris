
"""AG-heavy split sweep. The trace shows 41% of the kernel is an AllGather-only
tail with an idle fabric, so AG may want MORE workgroups than RS despite the
contended phase preferring fewer. Every split measured so far has AG <= RS, so
this is untested. Measured, not predicted."""
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, socket
N_GLOBAL, K_GLOBAL = 2880, 4096
WARMUP, ITERS = 30, 50

def bench(fn):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS

def _worker(local_rank, world_size, init_url):
    dist.init_process_group(backend="nccl", init_method=init_url,
                            world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    shmem = iris.iris(1 << 33); rank = shmem.get_rank()
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer, matmul_all_reduce_hbm_buffer_preamble)
    dt = torch.float16; KL = K_GLOBAL // world_size
    def P(*a):
        if rank == 0: print(*a, flush=True)
    for M, bm, bn, mf, tpf in [(2048, 128, 128, 16, 2)]:
        A = shmem.zeros((M, KL), device="cuda", dtype=dt)
        A.copy_(torch.randn(M, KL, dtype=dt, device=f"cuda:{rank}") * 0.1)
        B = torch.randn(KL, N_GLOBAL, dtype=dt, device=f"cuda:{rank}") * 0.1
        ref = torch.mm(A, B); dist.all_reduce(ref); torch.cuda.synchronize()
        out = torch.zeros(M, N_GLOBAL, device=f"cuda:{rank}", dtype=dt)
        tmp = torch.zeros_like(out)
        t = bench(lambda: (torch.mm(A, B, out=tmp), dist.all_reduce(tmp)))
        P(f"\n=== M={M} bm={bm} bn={bn} mfma={mf} tpf={tpf}   torch={t:.4f} ms ===")
        P(f"{'G/R/A':>12} {'ms':>9} {'vs torch':>9} {'maxdiff':>9} {'':>6}")
        splits = [(208,32,16,False),(208,32,16,True),(192,32,32,False),(192,32,32,True),(224,16,16,True),(208,16,32,True)]
        best = (9e9, None)
        for (g, r, a, th) in splits:
            try:
                # fresh workspace per config -- never reuse, the counters are monotonic
                ws = matmul_all_reduce_hbm_buffer_preamble(shmem, M, N_GLOBAL, dt, bm, bn)
                shmem.barrier()
                kw = dict(block_m=bm, block_n=bn, block_k=64, num_gemm_sms=g,
                          num_rs_sms=r, num_ag_sms=a, mfma=mf, tiles_per_flag=tpf, tail_help=th)
                ok = True; d = 0.0
                for _ in range(3):
                    out.zero_()
                    matmul_all_reduce_hbm_buffer(shmem, out, A, B, workspace=ws, **kw)
                    torch.cuda.synchronize()
                    d = torch.abs(out - ref).max().item()
                    if d > 0.05: ok = False; break
                shmem.barrier()
                if not ok:
                    P(f"{f"{g}/{r}/{a}{'+T' if th else ''}":>12} {'--':>9} {'--':>9} {d:>9.4f}  FAIL"); continue
                ms = bench(lambda: matmul_all_reduce_hbm_buffer(shmem, out, A, B,
                                                                workspace=ws, **kw))
                tag = ""
                if ms < best[0]: best = (ms, f"{g}/{r}/{a}"); tag = "  <-- best"
                P(f"{f"{g}/{r}/{a}{'+T' if th else ''}":>12} {ms:>9.4f} {t/ms:>8.2f}x {d:>9.4f} {tag}")
            except Exception as ex:
                P(f"{f"{g}/{r}/{a}{'+T' if th else ''}":>12}  ERR {type(ex).__name__}: {str(ex)[:44]}")
        P(f"  best {best[1]} at {best[0]:.4f} ms = {t/best[0]:.2f}x torch")
    dist.destroy_process_group()

def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("-r", "--num_ranks", type=int, default=8)
    a = p.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(fn=_worker, args=(a.num_ranks, f"tcp://127.0.0.1:{port}"),
             nprocs=a.num_ranks, join=True)

if __name__ == "__main__":
    main()
