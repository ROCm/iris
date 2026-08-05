'''What efficiency does RCCL AR actually achieve? Sets the ceiling for any two-shot.'''
import os, torch, torch.distributed as dist
lr=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(lr)
dist.init_process_group(backend='nccl')
ws,rank=dist.get_world_size(),dist.get_rank(); DT=torch.float16
def P(*a):
    if rank==0: print(*a,flush=True)
def bench(f,n=200,w=50):
    for _ in range(w): f()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record();torch.cuda.synchronize();return s.elapsed_time(e)/n
N,K=2880,4096; KL=K//ws
P(f"{'M':>6} {'AR-only ms':>11} {'algoMB':>8} {'busGB/s':>9} {'GEMM ms':>9} {'mm+AR ms':>9}")
for M in [128,512,2048]:
    X=torch.randn(M,N,device=f'cuda:{rank}',dtype=DT)
    A=torch.randn(M,KL,device=f'cuda:{rank}',dtype=DT)
    B=torch.randn(KL,N,device=f'cuda:{rank}',dtype=DT)
    C=torch.empty(M,N,device=f'cuda:{rank}',dtype=DT)
    ar=bench(lambda:dist.all_reduce(X))
    gm=bench(lambda:torch.mm(A,B,out=C))
    both=bench(lambda:(torch.mm(A,B,out=C),dist.all_reduce(C)))
    algo=2*(ws-1)/ws*M*N*2
    P(f'{M:>6} {ar:>11.4f} {algo/1e6:>8.2f} {algo/(ar*1e-3)/1e9:>9.1f} {gm:>9.4f} {both:>9.4f}')
dist.destroy_process_group()
