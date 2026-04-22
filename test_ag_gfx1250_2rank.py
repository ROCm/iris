"""Test gfx1250 async copy kernel with 2 simulated ranks on 1 GPU."""
import os

os.environ["IRIS_SIMULATION"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group(backend="gloo", rank=0, world_size=1)

import iris.experimental.iris_gluon as ig
from iris.ccl.config import Config

# Force 2 simulated ranks on the same GPU
ctx = ig.iris(heap_size=1 << 28)
rank = ctx.get_rank()
world_size = ctx.get_num_ranks()
print(f"rank={rank}, world_size={world_size}")

if world_size < 2:
    print("world_size=1, cannot test remote stores. Need torchrun --nproc_per_node=2")
    print("Testing local-only path instead...")

M, N = 64, 64
t_in = ctx.zeros((M, N), dtype=torch.float16)
t_in.fill_(float(rank + 1))
t_out = ctx.zeros((world_size * M, N), dtype=torch.float16)

config = Config(use_gluon=True, block_size_m=32, block_size_n=64, comm_sms=4)
print(f"config: block_size={config.block_size_m * config.block_size_n}, "
      f"threads_per_warp={config.threads_per_warp}, num_warps={config.num_warps}")

ctx.barrier()
ctx.ccl.all_gather(t_out, t_in, config=config)
torch.cuda.synchronize()

# Validate
passed = True
for r in range(world_size):
    expected = float(r + 1)
    chunk = t_out[r * M : (r + 1) * M]
    if not torch.allclose(chunk, torch.full_like(chunk, expected), atol=0.5):
        print(f"FAIL: chunk {r} got {chunk[0, 0].item():.1f}, expected {expected:.1f}")
        passed = False

if passed:
    print(f"PASS: out[0,0]={t_out[0, 0].item():.1f}")
else:
    print("FAIL")

dist.destroy_process_group()
