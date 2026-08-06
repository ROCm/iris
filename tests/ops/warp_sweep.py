'''Sweep num_warps on the HBM-buffer fused two-shot AR (never swept before).'''
import torch, torch.distributed as dist, torch.multiprocessing as mp, iris, socket
N_GLOBAL, K_GLOBAL = 2880, 4096
WARMUP, ITERS = 20, 50
def bench(fn):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/ITERS
def _worker(local_rank, world_size, init_url):
    dist.init_process_group(backend='nccl',init_method=init_url,
                            world_size=world_size,rank=local_rank)
    torch.cuda.set_device(local_rank)
    shmem=iris.iris(1<<33); rank=shmem.get_rank()
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer, matmul_all_reduce_hbm_buffer_preamble)
    dt=torch.float16; KL=K_GLOBAL//world_size
    def P(*a):
        if rank==0: print(*a,flush=True)
    CFG=[(128,16,128,32,192,32,32,1),(512,64,128,16,96,96,64,1),(2048,128,128,16,96,96,64,2)]
    for (M,bm,bn,mf,G,R,Ag,tpf) in CFG:
        A=shmem.zeros((M,KL),device='cuda',dtype=dt)
        A.copy_(torch.randn(M,KL,dtype=dt,device=f'cuda:{rank}')*0.1)
        B=torch.randn(KL,N_GLOBAL,dtype=dt,device=f'cuda:{rank}')*0.1
        ref=torch.mm(A,B); dist.all_reduce(ref,op=dist.ReduceOp.SUM); torch.cuda.synchronize()
        out=torch.zeros(M,N_GLOBAL,device=f'cuda:{rank}',dtype=dt)
        tmp=torch.zeros_like(out)
        t=bench(lambda:(torch.mm(A,B,out=tmp),dist.all_reduce(tmp)))
        ws=matmul_all_reduce_hbm_buffer_preamble(shmem,M,N_GLOBAL,dt,bm,bn); shmem.barrier()
        P(f'\n=== M={M} bm={bm} bn={bn} mfma={mf} G/R/A={G}/{R}/{Ag} tpf={tpf}  torch={t:.4f}ms ===')
        P(f"{'warps':>6} {'ms':>9} {'vs torch':>9} {'vs w=8':>8} {'maxdiff':>9}")
        base=None
        for nw in [2,4,8,16]:
            kw=dict(block_m=bm,block_n=bn,block_k=64,num_gemm_sms=G,num_rs_sms=R,
                    num_ag_sms=Ag,mfma=mf,tiles_per_flag=tpf,num_warps=nw)
            try:
                ok=True; d=0.0
                for _ in range(3):
                    out.zero_()
                    matmul_all_reduce_hbm_buffer(shmem,out,A,B,workspace=ws,**kw)
                    torch.cuda.synchronize()
                    d=torch.abs(out-ref).max().item()
                    if d>0.05: ok=False; break
                shmem.barrier()
                if not ok:
                    P(f'{nw:>6} {"--":>9} {"--":>9} {"--":>8} {d:>9.4f} FAIL'); continue
                ms=bench(lambda kw=kw: matmul_all_reduce_hbm_buffer(shmem,out,A,B,workspace=ws,**kw))
                if nw==8: base=ms
                P(f'{nw:>6} {ms:>9.4f} {t/ms:>8.2f}x {(base/ms if base else float("nan")):>7.2f}x {d:>9.4f}')
            except Exception as ex:
                P(f'{nw:>6}   ERR {type(ex).__name__}: {str(ex)[:60]}')
    dist.destroy_process_group()
def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('-r','--num_ranks',type=int,default=8)
    p.add_argument('--port',type=int,default=0); a=p.parse_args()
    port=a.port
    if port==0:
        s=socket.socket(); s.bind(('127.0.0.1',0)); port=s.getsockname()[1]; s.close()
    mp.spawn(fn=_worker,args=(a.num_ranks,f'tcp://127.0.0.1:{port}'),nprocs=a.num_ranks,join=True)
if __name__=='__main__': main()
