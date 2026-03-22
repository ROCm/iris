#!/usr/bin/env python3
"""Dump generated AMDGCN assembly for Triton and Gluon all-gather kernels."""

import os
import sys
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")


def _worker(local_rank, world_size, init_url):
    dist.init_process_group(
        backend="nccl",
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    import iris
    import iris.experimental.iris_gluon as iris_gluon
    from iris.ccl import Config
    from iris.ccl.all_gather import all_gather

    # Use small tensors just to trigger compilation
    M, N = 128, 64
    dtype = torch.float16
    heap_size = 2**33

    # === Run Triton kernel to trigger compilation ===
    shmem_triton = iris.iris(heap_size)
    rank = shmem_triton.get_rank()
    ws = shmem_triton.get_num_ranks()

    inp = shmem_triton.zeros((M, N), dtype=dtype)
    inp.fill_(float(rank + 1))
    out = shmem_triton.zeros((ws * M, N), dtype=dtype)

    config_triton = Config()
    shmem_triton.barrier()
    all_gather(out, inp, shmem_triton, config=config_triton)
    shmem_triton.barrier()
    del shmem_triton
    import gc; gc.collect()

    if rank == 0:
        print("=== Triton kernel compiled ===")

    # === Run Gluon kernel to trigger compilation ===
    shmem_gluon = iris_gluon.iris(heap_size)

    inp_g = shmem_gluon.zeros((M, N), dtype=dtype)
    inp_g.fill_(float(rank + 1))
    out_g = shmem_gluon.zeros((ws * M, N), dtype=dtype)

    config_gluon = Config(use_gluon=True)
    shmem_gluon.barrier()
    all_gather(out_g, inp_g, shmem_gluon, config=config_gluon)
    shmem_gluon.barrier()
    del shmem_gluon
    gc.collect()

    if rank == 0:
        print("=== Gluon kernel compiled ===")

    # === Find and dump assembly from Triton cache ===
    if rank == 0:
        cache_dir = os.path.expanduser("~/.triton/cache")
        if "TRITON_CACHE_DIR" in os.environ:
            cache_dir = os.environ["TRITON_CACHE_DIR"]

        # Find all .amdgcn files
        amdgcn_files = glob.glob(os.path.join(cache_dir, "**", "*.amdgcn"), recursive=True)
        # Also check for .rocm files or .hsaco
        hsaco_files = glob.glob(os.path.join(cache_dir, "**", "*.hsaco"), recursive=True)
        # Check for assembly in various forms
        all_asm_files = glob.glob(os.path.join(cache_dir, "**", "*.*"), recursive=True)

        # Filter to just assembly-like files
        asm_extensions = {'.amdgcn', '.s', '.asm', '.hsaco', '.amdgcn_asm'}
        asm_files = [f for f in all_asm_files if os.path.splitext(f)[1] in asm_extensions]

        print(f"\n=== Triton cache dir: {cache_dir} ===")
        print(f"Total files in cache: {len(all_asm_files)}")
        print(f"ASM-like files found: {len(asm_files)}")

        if not asm_files:
            # List all unique extensions
            exts = set(os.path.splitext(f)[1] for f in all_asm_files)
            print(f"File extensions in cache: {exts}")

            # Look for files containing 'all_gather' in name
            ag_files = [f for f in all_asm_files if 'all_gather' in f.lower()]
            print(f"Files with 'all_gather' in path: {len(ag_files)}")
            for f in ag_files[:20]:
                print(f"  {f}")

            # Just list recent files (by mtime)
            recent = sorted(all_asm_files, key=os.path.getmtime, reverse=True)[:30]
            print(f"\nMost recent 30 files:")
            for f in recent:
                size = os.path.getsize(f)
                print(f"  {size:>8} {f}")

        for f in asm_files:
            size = os.path.getsize(f)
            # Check if it's related to all_gather by examining parent dirs
            print(f"\n{'='*60}")
            print(f"ASM file: {f} ({size} bytes)")
            if size < 500000:  # Don't dump huge files
                with open(f) as fh:
                    content = fh.read()
                    # Print first 200 lines
                    lines = content.split('\n')
                    for line in lines[:200]:
                        print(line)
                    if len(lines) > 200:
                        print(f"... ({len(lines) - 200} more lines)")

    dist.barrier()
    dist.destroy_process_group()


def main():
    mp.spawn(fn=_worker, args=(4, "tcp://127.0.0.1:29235"), nprocs=4, join=True)


if __name__ == "__main__":
    main()
