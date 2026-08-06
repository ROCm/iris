'''Isolate atomic-contention: atomic-poll vs load-poll vs load-poll+padded flags.'''
import os, torch, torch.distributed as dist, triton, triton.language as tl
import iris
lr=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(lr)
dist.init_process_group(backend='nccl')
ws,rank=dist.get_world_size(),dist.get_rank(); DT=torch.float16
def P(*a):
    if rank==0: print(*a,flush=True)

@triton.jit
def fused(A,B,C,out,flags,hb:tl.tensor,M,N,KL,
          sam,sak,sbk,sbn,scm,scn,som,son,target,
          cur:tl.constexpr,W:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
          GWG:tl.constexpr,TWG:tl.constexpr,SPIN:tl.constexpr,
          FSTRIDE:tl.constexpr, POLL_MODE:tl.constexpr):
    pid=tl.program_id(0)
    n_n=tl.cdiv(N,BN); n_t=tl.cdiv(M,BM)*n_n
    if pid<GWG:
        for t in range(pid,n_t,GWG):
            pm=t//n_n; pn=t%n_n
            rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
            mk=(rm[:,None]<M)&(rn[None,:]<N)
            acc=tl.zeros((BM,BN),dtype=tl.float32)
            for k0 in range(0,KL,BK):
                a=tl.load(A+rm[:,None]*sam+(k0+rk)[None,:]*sak,
                          mask=(rm[:,None]<M)&((k0+rk)[None,:]<KL),other=0.0)
                b=tl.load(B+(k0+rk)[:,None]*sbk+rn[None,:]*sbn,
                          mask=((k0+rk)[:,None]<KL)&(rn[None,:]<N),other=0.0)
                acc+=tl.dot(a,b)
            tl.store(C+rm[:,None]*scm+rn[None,:]*scn,acc.to(C.dtype.element_ty),
                     mask=mk,cache_modifier='.wt')
            fi=t*FSTRIDE
            tl.atomic_add(flags+fi,1,sem='release',scope='gpu')
            for p in tl.static_range(0,W):
                if p!=cur:
                    iris.atomic_add(flags+fi,1,cur,p,hb,sem='release',scope='sys')
    else:
        cid=pid-GWG; cwg=TWG-GWG
        for t in range(cid,n_t,cwg):
            fi=t*FSTRIDE; s=0
            if POLL_MODE==99:
                d=tl.atomic_add(flags+fi,0,sem='acquire',scope='gpu')
                while (d<target) and (s<SPIN):
                    d=tl.atomic_add(flags+fi,0,sem='acquire',scope='gpu'); s+=1
            else:
                d=tl.load(flags+fi,volatile=True)
                if POLL_MODE==2:
                    s=SPIN
                while (d<target) and (s<SPIN):
                    d=tl.load(flags+fi,volatile=True); s+=1
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
N,K=2880,4096; KL=K//ws; M=int(os.environ.get('TM','2048')); BM,BN=32,128
A=torch.randn(M,KL,device=f'cuda:{rank}',dtype=DT)
B=torch.randn(KL,N,device=f'cuda:{rank}',dtype=DT)
C=shmem.zeros((M,N),dtype=DT); out=torch.zeros(M,N,device=f'cuda:{rank}',dtype=DT)
Cr=torch.empty(M,N,device=f'cuda:{rank}',dtype=DT)
nt=((M+BM-1)//BM)*((N+BN-1)//BN)
t_ms=bench(lambda:(torch.mm(A,B,out=Cr),dist.all_reduce(Cr)))
torch.mm(A,B,out=C); ref=C.clone().float(); dist.all_reduce(ref)
P(f'M={M} tiles={nt} ws={ws}  torch={t_ms:.4f}ms')
P(f"{'poll':>8} {'fstride':>8} {'gemmWG':>7} {'commWG':>7} {'ms':>9} {'vs torch':>9} {'diff':>8}")
for mode,mname in [(1,'load-wait'),(2,'NO-WAIT')]:
    for fs in [1]:
        flags=shmem.zeros((nt*fs,),dtype=torch.int32)
        for g,c in [(128,64),(192,32)]:
            it=[0]
            def run(mode=mode,fs=fs,g=g,c=c,flags=flags):
                it[0]+=1
                fused[(g+c,)](A,B,C,out,flags,hb,M,N,KL,A.stride(0),A.stride(1),
                    B.stride(0),B.stride(1),C.stride(0),C.stride(1),
                    out.stride(0),out.stride(1),it[0]*ws,rank,ws,BM,BN,64,
                    g,g+c,200000,fs,mode)
            try:
                flags.zero_(); it[0]=0; shmem.barrier(); run(); torch.cuda.synchronize()
                d=(out.float()-ref).abs().max().item()
                if d>=1.0:
                    P(f'{mname:>8} {fs:>8} {g:>7} {c:>7} {"--":>9} {"--":>9} {d:>8.2f} FAIL'); continue
                ms=bench(run)
                P(f'{mname:>8} {fs:>8} {g:>7} {c:>7} {ms:>9.4f} {t_ms/ms:>8.2f}x {d:>8.4f}')
            except Exception as ex:
                P(f'{mname:>8} {fs:>8} {g:>7} {c:>7}  ERR {str(ex)[:38]}')
shmem.barrier(); dist.destroy_process_group()
