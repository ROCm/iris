#!/usr/bin/env python3
"""Push-based RS kernel — iris.store to peer instead of iris.load from peer.

At ws=2: rank 0 stores its chunk 1 to rank 1's staging, rank 1 stores its
chunk 0 to rank 0's staging. Then each rank locally sums own + received.

Also tests chunked pipelining (RCCL-style) to keep XGMI link saturated.
"""

import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)


@triton.jit
def push_reduce_scatter_kernel(
    input_ptr,
    staging_ptr,
    output_ptr,
    M,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    stride_stg_m,
    stride_stg_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Push-based RS:
    Phase 1: each rank iris.store its non-owned chunks to the owning rank's staging
    Phase 2: each rank locally sums own partial + received staging → output
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    num_m_tiles_local = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_local_tiles = num_m_tiles_local * num_n_tiles
    num_m_tiles = M // BLOCK_SIZE_M

    # Phase 1: Push non-owned chunks to peers via iris.store
    # At ws=2: rank 0 pushes chunk 1 (rows M_local..M) to rank 1's staging
    #          rank 1 pushes chunk 0 (rows 0..M_local) to rank 0's staging
    for peer in tl.static_range(world_size):
        if peer != cur_rank:
            peer_m_offset = peer * num_m_tiles_local

            for tile_id in range(pid, total_local_tiles, NUM_SMS):
                local_pid_m = tile_id // num_n_tiles
                pid_n = tile_id % num_n_tiles

                global_pid_m = peer_m_offset + local_pid_m

                rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

                # Read own partial for peer's chunk
                in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
                is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

                if is_full:
                    tile = tl.load(input_ptr + in_offset)
                    # Write to peer's staging buffer
                    out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    stg_offset = out_rm[:, None] * stride_stg_m + rn[None, :] * stride_stg_n
                    iris.store(staging_ptr + stg_offset, tile, cur_rank, peer, heap_bases)
                else:
                    mask = (rm[:, None] < M) & (rn[None, :] < N)
                    tile = tl.load(input_ptr + in_offset, mask=mask, other=0.0)
                    out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    stg_offset = out_rm[:, None] * stride_stg_m + rn[None, :] * stride_stg_n
                    out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                    iris.store(staging_ptr + stg_offset, tile, cur_rank, peer, heap_bases, mask=out_mask)

    # Barrier: ensure all pushes are complete before reduction
    # This is a device-side barrier — all WGs must reach here
    tl.debug_barrier()

    # Phase 2: Local sum — own partial (from input) + received (from staging)
    own_m_offset = cur_rank * num_m_tiles_local

    for tile_id in range(pid, total_local_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles

        global_pid_m = own_m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        stg_offset = out_rm[:, None] * stride_stg_m + rn[None, :] * stride_stg_n

        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

        if is_full:
            own = tl.load(input_ptr + in_offset).to(acc_dtype)
            received = tl.load(staging_ptr + stg_offset).to(acc_dtype)
            result = own + received
            out_offset = out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(output_ptr + out_offset, result.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            own = tl.load(input_ptr + in_offset, mask=mask, other=0.0).to(acc_dtype)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            received = tl.load(staging_ptr + stg_offset, mask=out_mask, other=0.0).to(acc_dtype)
            result = own + received
            out_offset = out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(output_ptr + out_offset, result.to(output_ptr.type.element_ty), mask=out_mask)


# --- Benchmark ---
M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 50, 200

input_tensor = shmem.zeros((M, N), dtype=dtype)
input_tensor.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
staging = shmem.zeros((M_local, N), dtype=dtype)
output_tensor = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

heap_bases = shmem.get_heap_bases()
shmem.barrier()

NUM_SMS = 304
configs = [(64, 64), (128, 64), (128, 128), (256, 64)]

if rank == 0:
    print(f"Push RS: M={M}, N={N}, TP={world_size}")

# RCCL baseline
input_rccl = input_tensor.clone()
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
end_evt.record()
torch.cuda.synchronize()
rccl_ms = start_evt.elapsed_time(end_evt) / iters

if rank == 0:
    print(f"RCCL RS: {rccl_ms:.3f}ms")

best_ms = 999.0
best_cfg = None

for bm, bn in configs:
    if M_local % bm != 0:
        continue
    out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    stg = shmem.zeros((M_local, N), dtype=dtype)

    shmem.barrier()

    for _ in range(warmup):
        stg.zero_()
        push_reduce_scatter_kernel[(NUM_SMS,)](
            input_tensor, stg, out,
            M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            stg.stride(0), stg.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size,
            bm, bn, NUM_SMS,
        )
    torch.cuda.synchronize()

    # Correctness
    ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    dist.reduce_scatter_tensor(ref, input_tensor.clone(), op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    max_diff = torch.abs(out - ref).max().item()

    if max_diff > 1.0:
        if rank == 0:
            print(f"  bm={bm} bn={bn}: FAIL (diff={max_diff:.2f})")
        continue

    start_evt.record()
    for _ in range(iters):
        stg.zero_()
        push_reduce_scatter_kernel[(NUM_SMS,)](
            input_tensor, stg, out,
            M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            stg.stride(0), stg.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size,
            bm, bn, NUM_SMS,
        )
    end_evt.record()
    torch.cuda.synchronize()

    ms = start_evt.elapsed_time(end_evt) / iters
    if rank == 0:
        bw = M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9
        print(f"  bm={bm:3d} bn={bn:3d}: {ms:.3f}ms ({bw:.1f} GB/s) diff={max_diff:.4f}")
    if ms < best_ms:
        best_ms = ms
        best_cfg = (bm, bn)

if rank == 0:
    print()
    print(f"RCCL RS:      {rccl_ms:.3f}ms")
    print(f"Push iris RS: {best_ms:.3f}ms (best: bm={best_cfg[0]} bn={best_cfg[1]})")
    print(f"Speedup:      {rccl_ms / best_ms:.2f}x")

shmem.barrier()
dist.destroy_process_group()
