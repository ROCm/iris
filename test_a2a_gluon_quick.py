#!/usr/bin/env python3
"""Quick test: does the all_to_all gluon kernel compile in this Triton version?"""
import os
import torch
import torch.distributed as dist
import iris.experimental.iris_gluon as iris_gluon
from iris.ccl import Config
from iris.ccl.all_to_all import all_to_all

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

shmem = iris_gluon.iris(2**30)
M, N = 128, 64
inp = shmem.zeros((M, N * world_size), dtype=torch.float16)
out = shmem.zeros((M, N * world_size), dtype=torch.float16)
inp.fill_(float(rank + 1))

shmem.barrier()
cfg = Config(use_gluon=True, block_size_n=N)
try:
    all_to_all(out, inp, shmem, config=cfg)
    shmem.barrier()
    if rank == 0:
        print("all_to_all gluon: PASS")
except Exception as e:
    if rank == 0:
        print(f"all_to_all gluon: FAIL - {e}")

dist.destroy_process_group()
