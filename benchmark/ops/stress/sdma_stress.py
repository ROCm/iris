"""Loop a GEMM+all-gather copy-engine op and fsync a progress line before every step.

If the host dies, the last line of each rank's progress log says exactly which iteration and phase
it was in. Run under torchrun:

    torchrun --nproc_per_node=2 benchmark/ops/stress/sdma_stress.py --mode host --iters 40 --tag t --logdir DIR

``run_ladder.sh`` and ``pair_sweep.sh`` in this directory drive it. Output of the last iteration is checked
against a dense matmul at the end.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

import iris
from tritonblas.matmul import _make_matmul_selector

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["host", "device"], required=True)
p.add_argument("--iters", type=int, default=40)
p.add_argument("--m_local", type=int, default=1024)
p.add_argument("--n", type=int, default=3584)
p.add_argument("--k", type=int, default=8192)
p.add_argument("--heap_gb", type=int, default=4)
p.add_argument("--tag", required=True)
p.add_argument("--logdir", required=True)
args = p.parse_args()

os.makedirs(args.logdir, exist_ok=True)
rank_env = int(os.environ.get("RANK", "0"))
logf = open(os.path.join(args.logdir, f"{args.tag}_{args.mode}_r{rank_env}.log"), "a", buffering=1)


def log(msg):
    logf.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    logf.flush()
    os.fsync(logf.fileno())


if args.mode == "host":
    from iris.ops.matmul_all_gather_host_copy_engine import (
        matmul_all_gather_host_copy_engine as op,
        matmul_all_gather_host_copy_engine_preamble as preamble,
    )
else:
    from iris.ops.matmul_all_gather_copy_engine import (
        matmul_all_gather_copy_engine as op,
        matmul_all_gather_copy_engine_preamble as preamble,
    )

log(f"start mode={args.mode} iters={args.iters} m_local={args.m_local} tag={args.tag}")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group(backend="nccl")
shmem = iris.iris(args.heap_gb << 30, engines=("sdma",))
rank, world = shmem.get_rank(), shmem.get_num_ranks()
dtype = torch.float16
device = torch.device(f"cuda:{torch.cuda.current_device()}")
selector = _make_matmul_selector(args.m_local, args.n, args.k, dtype, dtype, dtype, device, streamk=False)
torch.manual_seed(123 + rank)
A = shmem.randn((args.m_local, args.k), dtype=dtype)
torch.manual_seed(456)
B = shmem.randn((args.k, args.n), dtype=dtype)
C = shmem.zeros((args.m_local * world, args.n), dtype=dtype)
ws = (
    preamble(shmem, A, B, selector=selector)
    if args.mode == "device"
    else preamble(shmem, A, B, trace=False, selector=selector)
)
shmem.barrier()
log(f"ready world={world} tile={selector.block_m}x{selector.block_n}")

for i in range(args.iters):
    log(f"iter {i} before-op")
    kwargs = dict(async_op=False, workspace=ws, flag_iteration=i)
    if args.mode == "host":
        kwargs["trace"] = False
    op(shmem, C, A, B, **kwargs)
    log(f"iter {i} after-op (launched+waited)")
    torch.cuda.synchronize()
    log(f"iter {i} after-sync")

shmem.barrier()
# Final correctness check of the last iteration's gathered output.
ref = torch.matmul(A, B)
ok = True
for src in range(world):
    chunk = ref.clone() if rank == src else torch.empty_like(ref)
    dist.broadcast(chunk, src=src)
    ok &= torch.allclose(C[src * args.m_local : (src + 1) * args.m_local], chunk, atol=5e-2, rtol=5e-2)
log(f"done correct={ok}")
shmem.barrier()
