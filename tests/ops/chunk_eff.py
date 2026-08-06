'''Pull efficiency vs per-peer chunk size. Two-shot reads 8x smaller chunks
than one-shot; if efficiency tracks chunk size the two-shot deficit is structural.'''
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, triton, socket
import triton.language as tl
NG=2880
def bench(fn,n=100,w=25):
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    for _ in range(w): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/n
@triton.jit
def pull(C,outp,hb:tl.tensor,M,N,scm,scn,som,son,
         cur:tl.constexpr,W:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,WGS:tl.constexpr):
    pid=tl.program_id(0); n_n=tl.cdiv(N,BN); n_t=tl.cdiv(M,BM)*n_n
    for t in range(pid,n_t,WGS):
        pm=t//n_n; pn=t%n_n
        rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN)
        mk=(rm[:,None]<M)&(rn[None,:]<N); off=rm[:,None]*scm+rn[None,:]*scn
        acc=iris.load(C+off,cur,(cur+1)%W,hb,mask=mk).to(tl.float32)
        for i in tl.static_range(2,W+1):
            acc+=iris.load(C+off,cur,(cur+i)%W,hb,mask=mk).to(tl.float32)
        tl.store(outp+rm[:,None]*som+rn[None,:]*son,acc.to(outp.dtype.element_ty),mask=mk)
def _w(lr,W,url):
    dist.init_process_group(backend='nccl',init_method=url,world_size=W,rank=lr)
    torch.cuda.set_device(lr)
    sh=iris.iris(1<<33); hb=sh.get_heap_bases(); rank=sh.get_rank(); dt=torch.float16
    def P(*a):
        if rank==0: print(*a,flush=True)
    P(f"\n one-shot pull, W={W}. per-peer chunk = rows*{NG}*2 bytes")
    P(f"{'rows':>6} {'perPeerMB':>10} {'totalMB':>8} {'best ms':>9} {'GB/s':>8} {'%line':>6} {'cfg':>16}")
    for rows in [256,512,1024,2048,4096]:
        C=sh.randn((rows,NG),dtype=dt)
        o=torch.zeros(rows,NG,device=f'cuda:{rank}',dtype=dt)
        byts=W*rows*NG*2; pp=rows*NG*2
        best=(9e9,None)
        for BM,BN,wg,nw in [(32,64,64,8),(32,128,64,8),(16,128,128,8),(32,128,196,8),
                            (64,64,64,8),(32,64,128,8),(16,128,64,8),(64,128,64,8)]:
            if BM>rows: continue
            sh.barrier()
            ms=bench(lambda BM=BM,BN=BN,wg=wg,nw=nw: pull[(wg,)](C,o,hb,rows,NG,
                C.stride(0),C.stride(1),o.stride(0),o.stride(1),rank,W,BM,BN,wg,num_warps=nw))
            if ms<best[0]: best=(ms,f'{BM}x{BN} {wg}WG')
        gb=byts/(best[0]*1e-3)/1e9
        P(f'{rows:>6} {pp/1e6:>10.2f} {byts/1e6:>8.1f} {best[0]:>9.4f} {gb:>8.1f} {gb/448*100:>5.0f}% {best[1]:>16}')
    dist.destroy_process_group()
def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('-r','--num_ranks',type=int,default=8); a=p.parse_args()
    s=socket.socket(); s.bind(('127.0.0.1',0)); port=s.getsockname()[1]; s.close()
    mp.spawn(fn=_w,args=(a.num_ranks,f'tcp://127.0.0.1:{port}'),nprocs=a.num_ranks,join=True)
if __name__=='__main__': main()
