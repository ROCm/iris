'''Isolate the two-shot halves. RS reads M*N, AG reads (ws-1)/ws*M*N.
If both hit one-shot efficiency the fused kernel has fusion overhead;
if not, the two-shot access pattern itself is the limit.'''
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl
NG,KG=2880,4096
def bench(fn,n=100,w=25):
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    for _ in range(w): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n

@triton.jit
def rs_k(C,outp,hb:tl.tensor,M,N,scm,scn,som,son,
         cur:tl.constexpr,W:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,WGS:tl.constexpr):
    '''rank cur reduces its own row-shard by pulling that shard from all W peers.'''
    pid=tl.program_id(0); MS=M//W; n_n=tl.cdiv(N,BN); n_t=tl.cdiv(MS,BM)*n_n
    r0=cur*MS
    for t in range(pid,n_t,WGS):
        pm=t//n_n; pn=t%n_n
        rm=r0+pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN)
        mk=(rm[:,None]<r0+MS)&(rn[None,:]<N); off=rm[:,None]*scm+rn[None,:]*scn
        # stagger the peer walk per workgroup so all WGs are not hammering
        # the same peer link at the same instant
        s0=(cur+1+pid)%W
        acc=iris.load(C+off,cur,s0,hb,mask=mk).to(tl.float32)
        for i in tl.static_range(1,W):
            acc+=iris.load(C+off,cur,(s0+i)%W,hb,mask=mk).to(tl.float32)
        om=pm*BM+tl.arange(0,BM)
        tl.store(outp+om[:,None]*som+rn[None,:]*son,acc.to(outp.dtype.element_ty),
                 mask=(om[:,None]<MS)&(rn[None,:]<N))

@triton.jit
def ag_k(S,outp,hb:tl.tensor,M,N,ssm,ssn,som,son,
         cur:tl.constexpr,W:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,WGS:tl.constexpr,
         INTERLEAVE:tl.constexpr=False):
    '''rank cur gathers the W-1 peer shards into the full [M,N] output.'''
    pid=tl.program_id(0); MS=M//W; n_n=tl.cdiv(N,BN); n_pt=tl.cdiv(MS,BM)*n_n
    n_t=n_pt*(W-1)
    for t in range(pid,n_t,WGS):
        if INTERLEAVE:
            pk=t%(W-1); lt=t//(W-1)
        else:
            pk=t//n_pt; lt=t%n_pt
        src=(cur+1+pk)%W
        pm=lt//n_n; pn=lt%n_n
        rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN)
        mk=(rm[:,None]<MS)&(rn[None,:]<N)
        v=iris.load(S+rm[:,None]*ssm+rn[None,:]*ssn,cur,src,hb,mask=mk)
        om=src*MS+pm*BM+tl.arange(0,BM)
        tl.store(outp+om[:,None]*som+rn[None,:]*son,v,mask=mk)

def _w(lr,W,url):
    dist.init_process_group(backend='nccl',init_method=url,world_size=W,rank=lr)
    torch.cuda.set_device(lr)
    sh=iris.iris(1<<33); hb=sh.get_heap_bases(); rank=sh.get_rank(); dt=torch.float16
    def P(*a):
        if rank==0: print(*a,flush=True)
    for M in [2048]:
        MS=M//W
        C=sh.randn((M,NG),dtype=dt)
        S=sh.randn((MS,NG),dtype=dt)
        rso=torch.zeros(MS,NG,device=f'cuda:{rank}',dtype=dt)
        ago=torch.zeros(M,NG,device=f'cuda:{rank}',dtype=dt)
        rs_b=M*NG*2; ag_b=(W-1)*MS*NG*2
        P(f'\n=== M={M} ws={W}  RS reads {rs_b/1e6:.1f}MB  AG reads {ag_b/1e6:.1f}MB  '
          f'total {(rs_b+ag_b)/1e6:.1f}MB ===')
        P(f'  RCCL AR moves 20.64MB in 0.0979ms = 210.9 GB/s. 2x target = 0.049ms total.')
        sh.barrier()
        rs_k[(196,)](C,rso,hb,M,NG,C.stride(0),C.stride(1),rso.stride(0),rso.stride(1),
                     rank,W,32,128,196,num_warps=8); torch.cuda.synchronize()
        full=C.clone(); dist.all_reduce(full)
        rs_d=(rso.float()-full[rank*MS:(rank+1)*MS].float()).abs().max().item()
        sh.barrier()
        ag_k[(196,)](S,ago,hb,M,NG,S.stride(0),S.stride(1),ago.stride(0),ago.stride(1),
                     rank,W,32,128,196,True,num_warps=8); torch.cuda.synchronize()
        gath=[torch.empty_like(S) for _ in range(W)]
        dist.all_gather(gath,S)
        ag_ref=torch.cat(gath,0)
        # AG never writes its OWN shard - the caller already has it locally
        ago[rank*MS:(rank+1)*MS].copy_(S)
        ag_d=(ago.float()-ag_ref.float()).abs().max().item()
        verdict = "PASS" if max(rs_d,ag_d)<0.05 else "FAIL"
        P(f'  correctness: RS max_diff={rs_d:.5f}  AG-interleave max_diff={ag_d:.5f}  {verdict}')
        sh.barrier()
        for nm,fn,byts,args in [('AG-interleave',ag_k,ag_b,(S,ago,hb,M,NG,S.stride(0),S.stride(1),ago.stride(0),ago.stride(1))),('RS',rs_k,rs_b,(C,rso,hb,M,NG,C.stride(0),C.stride(1),rso.stride(0),rso.stride(1))),
                                ('AG',ag_k,ag_b,(S,ago,hb,M,NG,S.stride(0),S.stride(1),ago.stride(0),ago.stride(1)))]:
            P(f"  {nm}: {'BM':>4} {'BN':>4} {'WG':>4} {'w':>2} {'ms':>9} {'GB/s':>8} {'%line':>6}")
            best=(9e9,None)
            for BM,BN,wg,nw in [(32,128,64,8),(32,128,64,4),(32,128,128,8),(32,64,64,8),
                                (16,128,64,8),(16,256,64,8),(32,256,64,8),(64,128,64,8),
                                (32,128,32,8),(32,128,196,8),(16,128,128,8),(64,64,64,8)]:
                sh.barrier()
                il=(nm=='AG-interleave')
                ms=bench(lambda BM=BM,BN=BN,wg=wg,nw=nw,il=il: fn[(wg,)](*args,rank,W,BM,BN,wg,il,num_warps=nw) if fn is ag_k else fn[(wg,)](*args,rank,W,BM,BN,wg,num_warps=nw))
                gb=byts/(ms*1e-3)/1e9
                if ms<best[0]: best=(ms,f'{BM}x{BN} {wg}WG {nw}w')
                P(f'       {BM:>4} {BN:>4} {wg:>4} {nw:>2} {ms:>9.4f} {gb:>8.1f} {gb/448*100:>5.0f}%')
            P(f'  >> {nm} best {best[0]:.4f}ms ({best[1]}) = {byts/(best[0]*1e-3)/1e9:.1f} GB/s')
    dist.destroy_process_group()
def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('-r','--num_ranks',type=int,default=8); a=p.parse_args()
    s=socket.socket(); s.bind(('127.0.0.1',0)); port=s.getsockname()[1]; s.close()
    mp.spawn(fn=_w,args=(a.num_ranks,f'tcp://127.0.0.1:{port}'),nprocs=a.num_ranks,join=True)
if __name__=='__main__': main()
