#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import random
import numpy as np

import iris

torch.manual_seed(123)
random.seed(123)


@triton.jit
def needleman_wunsch_kernel(seq1_ptr, seq2_ptr, dp_ptr, output_ptr, M,  N,match: tl.constexpr, mismatch: tl.constexpr, gap: tl.constexpr,cur_rank: tl.constexpr,world_size: tl.constexpr,BLOCK_SIZE: tl.constexpr,heap_bases_ptr: tl.tensor,
):
    pid = tl.program_id(0)
    
    seq1_off = pid * M
    seq2_off = pid * N
    dp_off = pid * (M + 1) * (N + 1)
    
    # Initialize dp table 
    for j in range(N + 1):
        offset = tl.full([1], dp_off + j, dtype=tl.int64)
        mask = tl.full([1], True, dtype=tl.int1)
        value = tl.full([1], -j * gap, dtype=tl.float32)
        iris.store(dp_ptr + offset, value, cur_rank, cur_rank, heap_bases_ptr, mask=mask)
    for i in range(1, M + 1):
        offset = tl.full([1], dp_off + i * (N + 1), dtype=tl.int64)
        mask = tl.full([1], True, dtype=tl.int1)
        value = tl.full([1], -i * gap, dtype=tl.float32)
        iris.store(dp_ptr + offset, value, cur_rank, cur_rank, heap_bases_ptr, mask=mask)
    #filll
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            mask = tl.full([1], True, dtype=tl.int1)
            
            s1_offset = tl.full([1], seq1_off + (i - 1), dtype=tl.int64)
            s2_offset = tl.full([1], seq2_off + (j - 1), dtype=tl.int64)
            s1 = iris.load(seq1_ptr + s1_offset, cur_rank, cur_rank, heap_bases_ptr, mask=mask)
            s2 = iris.load(seq2_ptr + s2_offset, cur_rank, cur_rank, heap_bases_ptr, mask=mask)
            
            match_score = tl.where(s1 == s2, match, mismatch)
            
            diag_offset = tl.full([1], dp_off + (i-1) * (N+1) + (j-1), dtype=tl.int64)
            up_offset = tl.full([1], dp_off + (i-1) * (N+1) + j, dtype=tl.int64)
            left_offset = tl.full([1], dp_off + i * (N+1) + (j-1), dtype=tl.int64)
            
            diag_score = iris.load(dp_ptr + diag_offset, cur_rank, cur_rank, heap_bases_ptr, mask=mask) + match_score
            up_score = iris.load(dp_ptr + up_offset, cur_rank, cur_rank, heap_bases_ptr, mask=mask) + gap
            left_score = iris.load(dp_ptr + left_offset, cur_rank, cur_rank, heap_bases_ptr, mask=mask) + gap
            
            max_score = tl.maximum(diag_score, tl.maximum(up_score, left_score))
            curr_offset = tl.full([1], dp_off + i * (N+1) + j, dtype=tl.int64)
            iris.store(dp_ptr + curr_offset, max_score, cur_rank, cur_rank, heap_bases_ptr, mask=mask)
    
    mask = tl.full([1], True, dtype=tl.int1)
    final_offset = tl.full([1], dp_off + M * (N+1) + N, dtype=tl.int64)
    final_score = iris.load(dp_ptr + final_offset, cur_rank, cur_rank, heap_bases_ptr, mask=mask)
    output_offset = tl.full([1], pid, dtype=tl.int64)
    iris.store(output_ptr + output_offset, final_score, cur_rank, cur_rank, heap_bases_ptr, mask=mask)


def bench_needleman_wunsch(shmem, batch_size, seq_len, num_experiments, num_warmup):

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    seq1 = shmem.randint(0, 4, (batch_size, seq_len), device="cuda", dtype=torch.int32)
    seq2 = shmem.randint(0, 4, (batch_size, seq_len), device="cuda", dtype=torch.int32)
    dp = shmem.zeros((batch_size, (seq_len+1) * (seq_len+1)), device="cuda", dtype=torch.float32)
    output = shmem.zeros(batch_size, device="cuda", dtype=torch.float32)
    
    BLOCK_SIZE = 256
    
    def run_kernel():
        grid = lambda meta: (batch_size,)  
        needleman_wunsch_kernel[grid](
            seq1, seq2, dp, output,
            seq_len, seq_len,
            1, -1, -2,  # standard match, mismatch, gap valuees
            rank, world_size, BLOCK_SIZE, heap_bases
        )
    
    timing_ms = iris.do_bench(run_kernel, shmem.barrier, n_repeat=num_experiments, n_warmup=num_warmup)
    
    total_ops = batch_size * seq_len * seq_len
    throughput = total_ops / (timing_ms * 1e-3)
    
    return timing_ms, throughput


## copied from other eg. (00_load/load_bench.py)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Needleman-Wunsch sequence alignment benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Iris heap size")
    parser.add_argument("--num_experiments", type=int, default=20, help="Number of experiments")
    parser.add_argument("--num_warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("-r", "--num_ranks", type=int, default=2, help="Number of ranks")
    
    return vars(parser.parse_args())


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    """Worker function for PyTorch distributed execution"""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=init_url, world_size=world_size, rank=local_rank)
    
    torch.cuda.set_device(local_rank)
    device_name = torch.cuda.get_device_name(local_rank)
    
    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    print(f"Rank {rank}/{world_size-1} initialized on GPU {local_rank}: {device_name}") # printing to make sure how many gpus used since gpu * 8 not available on cloud droplt
    dist.barrier()  
    
    timing_ms, throughput = bench_needleman_wunsch(
        shmem, 
        args["batch_size"], 
        args["seq_len"],
        args["num_experiments"],
        args["num_warmup"]
    )
    
    if rank == 0:
        print(f"\nNeedleman-Wunsch benchmark:")
        print(f"Batch size: {args['batch_size']}, Seq length: {args['seq_len']}")
        print(f"World size: {world_size} GPUs")
        print(f"Timing: {timing_ms:.3f} ms")
        print(f"Throughput: {throughput/1e6:.2f} M ops/s")
    
    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]
    init_url = "tcp://127.0.0.1:29500"
    
    mp.spawn(
        fn=_worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()