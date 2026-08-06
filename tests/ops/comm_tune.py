"""Comm-side tuning on the two-kernel winner at M=512/2048 (past crossover)."""
import os, torch, torch.distributed as dist, triton, triton.language as tl
import iris
lr=int(os.environ.get("LOCAL_RANK",0)); torch.cuda.set_device(lr)
dist.init_process_group(backend="nccl")
ws,rank=dist.get_world_size(),dist.get_rank(); DT=torch.float16
def P(*a):
    if rank==0: print(*a,flush=True)

@triton.jit
def one_shot(inp,out,hb:tl.tensor,n,cur:tl.constexpr,W:tl.constexpr,
             BLOCK:tl.constexpr,NS:tl.constexpr):
    pid=tl.program_id(0)
    for b in range(pid,tl.cdiv(n,BLOCK),NS):
        o=b*BLOCK+tl.arange(0,BLOCK); m=o<n
        sr=(cur+1)%W
        acc=iris.load(inp+o,cur,sr,hb,mask=m).to(tl.float32)
        for i in tl.static_range(1,W):
            acc+=iris.load(inp+o,cur,(sr+i)%W,hb,mask=m).to(tl.float32)
        tl.store(out+o,acc.to(out.dtype.element_ty),mask=m)

def bench(f,n=250,w=50):
    for _ in range(w): f()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/n

shmem=iris.iris(1<<33); hb=shmem.get_heap_bases()
N,K=2880,4096; KL=K//ws
MAXCU=torch.cuda.get_device_properties(rank).multi_processor_count
P(f"comm tune, two-kernel one-shot AR. ws={ws} CUs={MAXCU}")
for M in [256,512,1024,2048]:
    A=torch.randn(M,KL,device=f"cuda:{rank}",dtype=DT)
    B=torch.randn(KL,N,device=f"cuda:{rank}",dtype=DT)
    n=M*N; C=shmem.zeros((M,N),dtype=DT)
    out=torch.zeros(M,N,device=f"cuda:{rank}",dtype=DT)
    Cr=torch.empty(M,N,device=f"cuda:{rank}",dtype=DT)
    t_ms=bench(lambda:(torch.mm(A,B,out=Cr),dist.all_reduce(Cr)))
    res=[]
    for BL in [512,1024,2048,4096,8192]:
        for nc in [64,128,196,MAXCU]:
            for wp in [2,4,8]:
                try:
                    ms=bench(lambda BL=BL,nc=nc,wp=wp:(torch.mm(A,B,out=C),
                        one_shot[(nc,)](C.view(-1),out.view(-1),hb,n,rank,ws,BL,nc,num_warps=wp)),
                        n=150,w=30)
                    res.append((ms,BL,nc,wp))
                except Exception: pass
    res.sort()
    ms,BL,nc,wp=res[0]
    P(f"M={M:>5} torch={t_ms:.4f} best={ms:.4f} ({t_ms/ms:.2f}x) BLOCK={BL} sms={nc} w={wp}")
shmem.barrier(); dist.destroy_process_group()
