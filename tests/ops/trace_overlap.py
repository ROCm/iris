"""Trace GEMM/comm overlap in fused WG-spec GEMM+AR. Exports Perfetto JSON."""
import os, json, torch, torch.distributed as dist, triton, triton.language as tl
import iris
from iris.host.tracing.events import TraceEvent
from iris.mem.triton.context import Context as DeviceContext

lr=int(os.environ.get("LOCAL_RANK",0)); torch.cuda.set_device(lr)
dist.init_process_group(backend="nccl")
ws,rank=dist.get_world_size(),dist.get_rank(); DT=torch.float16
def P(*a):
    if rank==0: print(*a,flush=True)

@triton.jit
def fused_traced(A,B,C,out,flags,hb:tl.tensor,context_tensor,M,N,KL,
                 sam,sak,sbk,sbn,scm,scn,som,son,target,
                 cur:tl.constexpr,W:tl.constexpr,
                 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                 GWG:tl.constexpr,TWG:tl.constexpr,SPIN:tl.constexpr,
                 TRACE:tl.constexpr):
    ctx=DeviceContext.initialize(context_tensor,cur,W,tracing=TRACE)
    pid=tl.program_id(0)
    n_n=tl.cdiv(N,BN); n_t=tl.cdiv(M,BM)*n_n
    if pid < GWG:
        for t in range(pid,n_t,GWG):
            pm=t//n_n; pn=t%n_n
            rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
            mk=(rm[:,None]<M)&(rn[None,:]<N)
            # trace the GEMM tile as a "store" event
            h=ctx.tracing.record_event_start(event_id=TraceEvent().store,
                target_rank=cur, address=C+rm[:,None]*scm+rn[None,:]*scn,
                pid_m=pm, pid_n=pn, mask=mk)
            acc=tl.zeros((BM,BN),dtype=tl.float32)
            for k0 in range(0,KL,BK):
                a=tl.load(A+rm[:,None]*sam+(k0+rk)[None,:]*sak,
                          mask=(rm[:,None]<M)&((k0+rk)[None,:]<KL),other=0.0)
                b=tl.load(B+(k0+rk)[:,None]*sbk+rn[None,:]*sbn,
                          mask=((k0+rk)[:,None]<KL)&(rn[None,:]<N),other=0.0)
                acc+=tl.dot(a,b)
            tl.store(C+rm[:,None]*scm+rn[None,:]*scn,acc.to(C.dtype.element_ty),
                     mask=mk,cache_modifier=".wt")
            ctx.tracing.record_event_end(h)
            tl.atomic_add(flags+t,1,sem="release",scope="gpu")
            for p in tl.static_range(0,W):
                if p!=cur:
                    iris.atomic_add(flags+t,1,cur,p,hb,sem="release",scope="sys")
    else:
        cid=pid-GWG; cwg=TWG-GWG
        for t in range(cid,n_t,cwg):
            s=0
            d=tl.atomic_add(flags+t,0,sem="acquire",scope="gpu")
            while (d<target) and (s<SPIN):
                d=tl.atomic_add(flags+t,0,sem="acquire",scope="gpu"); s+=1
            pm=t//n_n; pn=t%n_n
            rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN)
            mk=(rm[:,None]<M)&(rn[None,:]<N); off=rm[:,None]*scm+rn[None,:]*scn
            # trace the comm tile as a "load" event
            h=ctx.tracing.record_event_start(event_id=TraceEvent().load,
                target_rank=(cur+1)%W, address=C+off, pid_m=pm, pid_n=pn, mask=mk)
            sr=(cur+1)%W
            acc=iris.load(C+off,cur,sr,hb,mask=mk).to(tl.float32)
            for i in tl.static_range(1,W):
                acc+=iris.load(C+off,cur,(sr+i)%W,hb,mask=mk).to(tl.float32)
            ctx.tracing.record_event_end(h)
            tl.store(out+rm[:,None]*som+rn[None,:]*son,acc.to(out.dtype.element_ty),mask=mk)

shmem=iris.iris(1<<33); hb=shmem.get_heap_bases()
N,K=2880,4096; KL=K//ws
M=int(os.environ.get("TRACE_M","2048"))
GWG=int(os.environ.get("GWG","128")); CWG=int(os.environ.get("CWG","64"))
BM,BN=32,128
A=torch.randn(M,KL,device=f"cuda:{rank}",dtype=DT)
B=torch.randn(KL,N,device=f"cuda:{rank}",dtype=DT)
C=shmem.zeros((M,N),dtype=DT); out=torch.zeros(M,N,device=f"cuda:{rank}",dtype=DT)
nt=((M+BM-1)//BM)*((N+BN-1)//BN)
flags=shmem.zeros((nt,),dtype=torch.int32)

shmem.tracing.enable(max_events=2_000_000)
dctx=shmem.get_device_context()
P(f"tracing fused GEMM+AR: M={M} tiles={nt} gemmWG={GWG} commWG={CWG}")

# warmup untraced
for i in range(5):
    flags.zero_(); shmem.barrier()
    fused_traced[(GWG+CWG,)](A,B,C,out,flags,hb,dctx,M,N,KL,
        A.stride(0),A.stride(1),B.stride(0),B.stride(1),
        C.stride(0),C.stride(1),out.stride(0),out.stride(1),
        ws,rank,ws,BM,BN,64,GWG,GWG+CWG,200000,False)
torch.cuda.synchronize()

shmem.tracing.reset(); flags.zero_(); shmem.barrier()
fused_traced[(GWG+CWG,)](A,B,C,out,flags,hb,dctx,M,N,KL,
    A.stride(0),A.stride(1),B.stride(0),B.stride(1),
    C.stride(0),C.stride(1),out.stride(0),out.stride(1),
    ws,rank,ws,BM,BN,64,GWG,GWG+CWG,200000,True)
torch.cuda.synchronize(); shmem.barrier()
shmem.tracing.export("/workspace/iris/trace_fused.json", merge=True)
P(f"events recorded: {shmem.tracing.trace_counter.item()}")
shmem.barrier(); dist.destroy_process_group()
