import os

os.environ["IRIS_SIMULATION"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch
import iris.experimental.iris_gluon as ig
from iris.ccl.config import Config

print("imports ok")
ctx = ig.iris()
print(f"ctx ok, rank={ctx.rank()}, world={ctx.world_size()}")

t_in = torch.randn(64, 64, device="cuda", dtype=torch.float16)
t_out = torch.zeros(64 * ctx.world_size(), 64, device="cuda", dtype=torch.float16)
print("tensors ok, launching kernel...")

ctx.ccl.all_gather(t_out, t_in, config=Config(use_gluon=True))
print("kernel done")

ctx.barrier()
print("all good")
