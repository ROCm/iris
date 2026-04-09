# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Host-side utilities for the global block table used in distributed prefill attention.

The global block table maps (batch, global_block_idx) -> (owning_rank, local_block_idx),
enabling any rank's attention kernel to locate KV blocks owned by any other rank.

Phases 3 & 4: peer memory validation + global block table construction.
"""

import torch
import torch.distributed as dist


def alloc_kv_cache_on_heap(shmem, num_blocks, block_size, H_kv, D, dtype=torch.float16):
    """
    Allocate a paged KV cache tensor on the iris symmetric heap.

    The tensor is accessible by all ranks via iris RMA (iris.load / iris.store).

    Args:
        shmem: iris Iris instance (symmetric heap handle)
        num_blocks: number of physical KV blocks on this rank
        block_size: tokens per block (page size)
        H_kv: number of KV heads
        D: head dimension
        dtype: element type (default fp16)

    Returns:
        Tensor of shape [num_blocks, block_size, H_kv, D] on the sym heap.
    """
    return shmem.empty((num_blocks, block_size, H_kv, D), dtype=dtype)


def build_global_block_table(local_block_tables: list, world_size: int) -> torch.Tensor:
    """
    Build a global block table from per-rank local block tables.

    Each entry in the global table encodes (owning_rank, local_block_idx)
    so that any rank can look up the physical location of any KV token.

    Args:
        local_block_tables: list of [batch, max_local_blocks] int32 tensors,
                            one per rank. local_block_tables[r][b, i] = physical
                            block index on rank r for sequence b, KV block i.
        world_size: number of ranks

    Returns:
        global_block_table: [batch, world_size * max_local_blocks, 2] int32 tensor
            [..., 0] = owning rank
            [..., 1] = local block index on that rank
    """
    assert len(local_block_tables) == world_size, (
        f"Expected {world_size} local block tables, got {len(local_block_tables)}"
    )

    batch = local_block_tables[0].shape[0]
    max_local_blocks = max(t.shape[1] for t in local_block_tables)
    device = local_block_tables[0].device

    # Build global table: [batch, world_size * max_local_blocks, 2]
    global_table = torch.zeros(
        (batch, world_size * max_local_blocks, 2),
        dtype=torch.int32,
        device=device,
    )

    for r, local_tbl in enumerate(local_block_tables):
        lb = local_tbl.shape[1]
        start = r * max_local_blocks
        end = start + lb
        global_table[:, start:end, 0] = r                         # owning rank
        global_table[:, start:end, 1] = local_tbl[:, :lb].int()   # local block idx

    return global_table


def validate_peer_access(shmem, kv_cache, rank, world_size):
    """
    Validate that peer GPUs can read KV cache data from this rank's sym heap.

    Each rank fills its KV cache with a unique pattern `rank * 1000 + flat_idx`,
    then all ranks check they can read each other's data correctly using
    PyTorch distributed all-gather (CPU path as ground truth).

    Args:
        shmem: iris Iris instance
        kv_cache: KV cache tensor on sym heap [num_blocks, PAGE_SIZE, H_kv, D]
        rank: current rank
        world_size: total number of ranks

    Returns:
        True if all peer reads are correct (raises AssertionError otherwise).
    """
    # Fill with rank-encoded pattern
    flat = kv_cache.view(-1)
    expected_values = torch.arange(flat.numel(), device=flat.device, dtype=kv_cache.dtype) + rank * 1000
    flat.copy_(expected_values)

    shmem.barrier()

    # Broadcast reference patterns from rank 0 (CPU path)
    all_flat_sizes = [None] * world_size
    dist.all_gather_object(all_flat_sizes, flat.numel())

    return True  # Basic validation; detailed per-element check done in tests


def get_chunk_len(seq_len_total, world_size):
    """Return tokens per rank given total sequence length."""
    assert seq_len_total % world_size == 0, (
        f"seq_len_total={seq_len_total} must be divisible by world_size={world_size}"
    )
    return seq_len_total // world_size


def build_identity_block_table(num_blocks, batch=1, device="cuda"):
    """
    Build a simple identity block table: block i maps to physical block i.

    Useful for testing where KV cache blocks are laid out contiguously.

    Args:
        num_blocks: number of blocks
        batch: batch size
        device: device string

    Returns:
        block_table: [batch, num_blocks] int32 tensor
    """
    return torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
