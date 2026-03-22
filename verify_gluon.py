#!/usr/bin/env python3
"""Verify gluon all-gather correctness across all dtypes and sizes."""

import os
import torch
import torch.distributed as dist
import iris.experimental.iris_gluon as iris_gluon
from iris.ccl import Config
from iris.ccl.all_gather import all_gather

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris_gluon.iris(heap_size)

test_configs = [
    (256, 256, 32, 256, "spt=1"),
    (1024, 512, 32, 512, "spt=2"),
    (8192, 8192, 32, 1024, "spt=4"),
]

all_pass = True
for dtype in [torch.float16, torch.float32, torch.bfloat16]:
    for M, N, bm, bn, label in test_configs:
        inp = shmem.zeros((M, N), dtype=dtype)
        out = shmem.zeros((world_size * M, N), dtype=dtype)
        inp.fill_(float(rank + 1))

        # Reference
        ref_inp = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
        ref_out = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")
        ref_inp.fill_(float(rank + 1))
        shmem.barrier()
        dist.all_gather_into_tensor(ref_out, ref_inp)
        torch.cuda.synchronize()

        # Gluon
        out.zero_()
        shmem.barrier()
        config = Config(use_gluon=True, block_size_m=bm, block_size_n=bn)
        all_gather(out, inp, shmem, config=config)
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-3 if dtype == torch.float16 else 1e-5
        match = torch.allclose(out, ref_out, atol=atol)
        max_diff = torch.abs(out - ref_out).max().item()

        if not match:
            all_pass = False

        if rank == 0:
            status = "PASS" if match else "FAIL"
            dtype_str = str(dtype).replace("torch.", "")
            print(f"{label} {M}x{N} {dtype_str:8s} [{status}] max_diff={max_diff:.6f}")

if rank == 0:
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

dist.destroy_process_group()
