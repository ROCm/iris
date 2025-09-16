#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
from pathlib import Path
import iris
import sys

sys.path.insert(0, str(Path(__file__).parent / "../../examples/13_needleman_wunsch"))
import needleman_wunsch as iris_nw_module
sys.path.pop(0)

# Single-GPU Needleman-Wunsch kernel (copied from my bio-triton library) [https://github.com/DevManpreet5/Triton-Needleman-Wunsch]
@triton.jit
def bio_triton_needleman_wunsch_kernel(seq1_ptr, seq2_ptr, dp_ptr, output_ptr, M, N,
                           match: tl.constexpr, mismatch: tl.constexpr, gap: tl.constexpr):
    pid = tl.program_id(0)
    
    seq1_off = pid * M
    seq2_off = pid * N
    dp_off = pid * (M + 1) * (N + 1)
    
    for j in range(N + 1):
        tl.store(dp_ptr + dp_off + j, -j * gap)
    for i in range(1, M + 1):
        tl.store(dp_ptr + dp_off + i * (N + 1), -i * gap)
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            s1 = tl.load(seq1_ptr + seq1_off + (i - 1))
            s2 = tl.load(seq2_ptr + seq2_off + (j - 1))
            
            match_score = tl.where(s1 == s2, match, mismatch)
            
            diag_score = tl.load(dp_ptr + dp_off + (i-1) * (N+1) + (j-1)) + match_score
            up_score = tl.load(dp_ptr + dp_off + (i-1) * (N+1) + j) + gap
            left_score = tl.load(dp_ptr + dp_off + i * (N+1) + (j-1)) + gap
            
            max_score = tl.maximum(diag_score, tl.maximum(up_score, left_score))
            tl.store(dp_ptr + dp_off + i * (N+1) + j, max_score)
    
    final_score = tl.load(dp_ptr + dp_off + M * (N+1) + N)
    tl.store(output_ptr + pid, final_score)

def run_bio_triton_needleman_wunsch(seq1_batch, seq2_batch, match=1, mismatch=-1, gap=-2):
    batch_size, seq_len = seq1_batch.shape
    dp = torch.zeros((batch_size, (seq_len+1) * (seq_len+1)), device="cuda", dtype=torch.float32)
    output = torch.zeros(batch_size, device="cuda", dtype=torch.float32)
    grid = lambda meta: (batch_size,)
    bio_triton_needleman_wunsch_kernel[grid](
        seq1_batch, seq2_batch, dp, output,
        seq_len, seq_len, match, mismatch, gap
    )
    
    return output

# Needleman-Wunsch Torch (copied from my bio-triton library) [https://github.com/DevManpreet5/Triton-Needleman-Wunsch]
def needleman_wunsch_pytorch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    batch_size, M = seq1.shape
    N = seq2.shape[1]
    dp = torch.zeros((batch_size, M + 1, N + 1), device='cuda')
    dp[:, 0, :] = torch.arange(0, -(N+1)*gap, -gap).repeat(batch_size, 1)
    dp[:, :, 0] = torch.arange(0, -(M+1)*gap, -gap).unsqueeze(0).repeat(batch_size, 1)

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            match_score = (seq1[:, i-1] == seq2[:, j-1]).float() * match + (seq1[:, i-1] != seq2[:, j-1]).float() * mismatch
            dp[:, i, j] = torch.max(torch.stack([
                dp[:, i-1, j-1] + match_score,
                dp[:, i-1, j] + gap,
                dp[:, i, j-1] + gap
            ], dim=1), dim=1)[0]

    return dp[:, M, N]


def run_iris_needleman_wunsch(seq1_batch, seq2_batch, shmem, match=1, mismatch=-1, gap=-2):
    batch_size, seq_len = seq1_batch.shape
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    dp = shmem.zeros((batch_size, (seq_len+1) * (seq_len+1)), device="cuda", dtype=torch.float32)
    output = shmem.zeros(batch_size, device="cuda", dtype=torch.float32)
    
    BLOCK_SIZE = 256
    grid = lambda meta: (batch_size,) 
    iris_nw_module.needleman_wunsch_kernel[grid](
        seq1_batch, seq2_batch, dp, output,
        seq_len, seq_len, match, mismatch, gap,
        rank, world_size, BLOCK_SIZE, heap_bases
    )
    
    return output


def test_iris_vs_bio_triton_needleman_wunsch():
    heap_size = 1 << 30
    shmem = iris.iris(heap_size)
    batch_size, seq_len = 4, 16

    torch.manual_seed(42)
    seq1 = shmem.randint(0, 4, (batch_size, seq_len), device="cuda", dtype=torch.int32)
    seq2 = shmem.randint(0, 4, (batch_size, seq_len), device="cuda", dtype=torch.int32)

    seq1_regular = torch.tensor(seq1.cpu().numpy(), device="cuda", dtype=torch.int32)
    seq2_regular = torch.tensor(seq2.cpu().numpy(), device="cuda", dtype=torch.int32)

    iris_output = run_iris_needleman_wunsch(seq1, seq2, shmem, 1, -1, -2)
    bio_triton_output = run_bio_triton_needleman_wunsch(seq1_regular, seq2_regular, 1, -1, -2)

    torch.testing.assert_close(iris_output, bio_triton_output, rtol=1e-5, atol=1e-5)


def test_iris_vs_pytorch_correctness():
    heap_size = 1 << 30
    shmem = iris.iris(heap_size)
    batch_size, seq_len = 8, 32

    torch.manual_seed(42)
    seq1 = shmem.randint(0, 4, (batch_size, seq_len), device="cuda", dtype=torch.int32)
    seq2 = shmem.randint(0, 4, (batch_size, seq_len), device="cuda", dtype=torch.int32)

    seq1_regular = torch.tensor(seq1.cpu().numpy(), device="cuda", dtype=torch.int32)
    seq2_regular = torch.tensor(seq2.cpu().numpy(), device="cuda", dtype=torch.int32)

    iris_output = run_iris_needleman_wunsch(seq1, seq2, shmem, 1, -1, -2)
    pytorch_output = needleman_wunsch_pytorch(seq1_regular, seq2_regular, 1, -1, -2)

    torch.testing.assert_close(iris_output, pytorch_output, rtol=1e-5, atol=1e-5)




