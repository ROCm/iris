"""P2: unfused producer-consumer. 2 streams, GEMM and AR kernels run CONCURRENTLY.

GEMM kernel (stream A): tile -> C_sym (.wt) -> flag[t] += 1 on all ranks
AR   kernel (stream B): poll flag[t] == ws -> one-shot pull-reduce tile t

No static CU split inside one kernel: two independent launches, the HW scheduler
interleaves them. Sweep gemm_sms x ar_sms to control occupancy of each.
"""
import os, torch, torch.distributed as dist, triton, triton.language as tl
import iris

lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr)
dist.init_process_group(backend="nccl")
ws, rank = dist.get_world_size(), dist.get_rank()
DT = torch.float16
def P(*a):
    if rank == 0: print(*a, flush=True)

@triton.jit
def gemm_signal(A, B, C, flags, heap_bases: tl.tensor, M, N, K_local,
                sam, sak, sbk, sbn, scm, scn,
                cur_rank: tl.constexpr, world_size: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN); n_t = tl.cdiv(M, BM) * n_n
    for t in range(pid, n_t, NS):
        pm = t // n_n; pn = t % n_n
        rm = pm*BM + tl.arange(0, BM); rn = pn*BN + tl.arange(0, BN); rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K_local, BK):
            a = tl.load(A + rm[:,None]*sam + (k0+rk)[None,:]*sak,
                        mask=(rm[:,None]<M)&((k0+rk)[None,:]<K_local), other=0.0)
            b = tl.load(B + (k0+rk)[:,None]*sbk + rn[None,:]*sbn,
                        mask=((k0+rk)[:,None]<K_local)&(rn[None,:]<N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(C + rm[:,None]*scm + rn[None,:]*scn, acc.to(C.dtype.element_ty),
                 mask=(rm[:,None]<M)&(rn[None,:]<N), cache_modifier=".wt")
        tl.atomic_add(flags + t, 1, sem="release", scope="gpu")
        for p in tl.static_range(0, world_size):
            if p != cur_rank:
                iris.atomic_add(flags + t, 1, cur_rank, p, heap_bases, sem="release", scope="sys")

@triton.jit
def ar_consume(C, out, flags, heap_bases: tl.tensor, M, N, scm, scn, som, son, target,
               cur_rank: tl.constexpr, world_size: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, SPIN: tl.constexpr):
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN); n_t = tl.cdiv(M, BM) * n_n
    for t in range(pid, n_t, NS):
        s = 0
        d = tl.atomic_add(flags + t, 0, sem="acquire", scope="gpu")
        while (d < target) and (s < SPIN):
            d = tl.atomic_add(flags + t, 0, sem="acquire", scope="gpu"); s += 1
        pm = t // n_n; pn = t % n_n
        rm = pm*BM + tl.arange(0, BM); rn = pn*BN + tl.arange(0, BN)
        mk = (rm[:,None]<M)&(rn[None,:]<N); off = rm[:,None]*scm + rn[None,:]*scn
        sr = (cur_rank+1) % world_size
        acc = iris.load(C + off, cur_rank, sr, heap_bases, mask=mk).to(tl.float32)
        for i in tl.static_range(1, world_size):
            acc += iris.load(C + off, cur_rank, (sr+i)%world_size, heap_bases, mask=mk).to(tl.float32)
        tl.store(out + rm[:,None]*som + rn[None,:]*son, acc.to(out.dtype.element_ty), mask=mk)

def bench(fn, n=200, w=40):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n

shmem = iris.iris(1 << 33); hb = shmem.get_heap_bases()
N, K = 2880, 4096; KL = K // ws
MAXCU = torch.cuda.get_device_properties(rank).multi_processor_count
sA = torch.cuda.Stream(); sB = torch.cuda.Stream()
P(f"P2 producer-consumer (2 streams)  ws={ws} CUs={MAXCU}")

for M in [32, 128, 512, 2048]:
    A = torch.randn(M, KL, device=f"cuda:{rank}", dtype=DT)
    B = torch.randn(KL, N, device=f"cuda:{rank}", dtype=DT)
    C = shmem.zeros((M, N), dtype=DT)
    out = torch.zeros(M, N, device=f"cuda:{rank}", dtype=DT)
    Cr = torch.empty(M, N, device=f"cuda:{rank}", dtype=DT)
    torch_ms = bench(lambda: (torch.mm(A,B,out=Cr), dist.all_reduce(Cr)))
    torch.mm(A,B,out=C); ref = C.clone().float(); dist.all_reduce(ref)
    P(f"\n=== M={M}  torch={torch_ms:.4f}ms ===")
    best=(1e9,None)
    for BM,BN in [(32,128),(64,128),(128,128)]:
        if BM > M: continue
        nt = ((M+BM-1)//BM)*((N+BN-1)//BN)
        fl = shmem.zeros((nt,), dtype=torch.int32)
        for g in [64,128,192]:
            for c in [32,64,128]:
                it=[0]
                def run(BM=BM,BN=BN,g=g,c=c,fl=fl):
                    it[0]+=1
                    ev = torch.cuda.Event()
                    with torch.cuda.stream(sA):
                        gemm_signal[(g,)](A,B,C,fl,hb,M,N,KL,A.stride(0),A.stride(1),
                            B.stride(0),B.stride(1),C.stride(0),C.stride(1),rank,ws,BM,BN,64,g)
                        ev.record(sA)
                    with torch.cuda.stream(sB):
                        ar_consume[(c,)](C,out,fl,hb,M,N,C.stride(0),C.stride(1),
                            out.stride(0),out.stride(1),it[0]*ws,rank,ws,BM,BN,c,200000)
                    torch.cuda.current_stream().wait_stream(sA)
                    torch.cuda.current_stream().wait_stream(sB)
                try:
                    fl.zero_(); it[0]=0; shmem.barrier(); run(); torch.cuda.synchronize()
                    d=(out.float()-ref).abs().max().item()
                    if d>=1.0: continue
                    ms=bench(run,n=150,w=30)
                    if ms<best[0]: best=(ms,f"BM={BM} BN={BN} gemmSM={g} arSM={c}")
                except Exception: pass
    if best[1]:
        P(f"  BEST P2: {best[0]:.4f}ms ({torch_ms/best[0]:.2f}x vs torch)  {best[1]}")
    else:
        P("  no working config")
shmem.barrier(); dist.destroy_process_group()
