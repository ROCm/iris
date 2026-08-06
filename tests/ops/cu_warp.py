'''Does one-shot pull-AR scale with CU count? Comm only, no GEMM, no flags.'''
import os, torch, torch.distributed as dist, triton, triton.language as tl
import iris
lr=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(lr)
dist.init_process_group(backend='nccl')
ws,rank=dist.get_world_size(),dist.get_rank(); DT=torch.float16
def P(*a):
    if rank==0: print(*a,flush=True)
@triton.jit
def ar(C,out,hb:tl.tensor,M,N,scm,scn,som,son,
       cur:tl.constexpr,W:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,WGS:tl.constexpr):
    pid=tl.program_id(0); n_n=tl.cdiv(N,BN); n_t=tl.cdiv(M,BM)*n_n
    for t in range(pid,n_t,WGS):
        pm=t//n_n; pn=t%n_n
        rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN)
        mk=(rm[:,None]<M)&(rn[None,:]<N); off=rm[:,None]*scm+rn[None,:]*scn
        sr=(cur+1)%W
        acc=iris.load(C+off,cur,sr,hb,mask=mk).to(tl.float32)
        for i in tl.static_range(1,W):
            acc+=iris.load(C+off,cur,(sr+i)%W,hb,mask=mk).to(tl.float32)
        tl.store(out+rm[:,None]*som+rn[None,:]*son,acc.to(out.dtype.element_ty),mask=mk)
def bench(f,n=100,w=20):
    for _ in range(w): f()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/n
shmem=iris.iris(1<<33); hb=shmem.get_heap_bases()
N=2880
for M in [2048]:
    C=shmem.randn((M,N),dtype=DT); out=torch.zeros(M,N,device=f'cuda:{rank}',dtype=DT)
    byts=ws*M*N*2
    P(f'--- one-shot pull AR, M={M} N={N} ws={ws}: reads {byts/1e6:.1f} MB/rank ---')
    P(f"{'BM':>4} {'BN':>4} {'WG':>5} {'w':>2} {'ms':>9} {'GB/s':>9} {'% of 448':>9} {'vs 256WG':>9}")
    base=None
    for BM,BN,wg,nw in [(32,128,64,1),(32,128,64,2),(32,128,64,4),(32,128,64,8),(32,128,128,2),(32,128,196,2),(32,128,196,4),(32,128,304,2),(16,256,196,2),(16,256,304,2),(32,256,196,2),(32,256,304,4),(64,128,196,2),(16,512,196,4)]:
        shmem.barrier()
        ms=bench(lambda wg=wg,BM=BM,BN=BN,nw=nw: ar[(wg,)](C,out,hb,M,N,C.stride(0),C.stride(1),
            out.stride(0),out.stride(1),rank,ws,BM,BN,wg,num_warps=nw))
        gb=byts/(ms*1e-3)/1e9
        if base is None: base=ms
        P(f'{BM:>4} {BN:>4} {wg:>5} {nw:>2} {ms:>9.4f} {gb:>9.1f} {gb/448*100:>8.0f}% {(base/ms if base else float("nan")):>8.2f}x')
shmem.barrier(); dist.destroy_process_group()
