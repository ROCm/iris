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

print("imports ok")
ctx = ig.iris()
print(f"ctx ok, rank={ctx.get_rank()}, world={ctx.get_num_ranks()}")

world_size = ctx.get_num_ranks()
t_in = ctx.zeros((64, 64), dtype=torch.float16)
t_in.fill_(float(ctx.get_rank() + 1))
t_out = ctx.zeros((64 * world_size, 64), dtype=torch.float16)
print("tensors ok, launching kernel...")

ctx.ccl.all_gather(t_out, t_in, config=Config(use_gluon=True))
print("kernel done")

ctx.barrier()
print("all good")
