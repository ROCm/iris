# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Iris device kernels driving rocSHMEM-allocated buffers.

Intra-node (IPC). Run on one node with 2+ ranks:

  torchrun --nproc_per_node=2 tests/manual_rocshmem_provider.py

Set EXPECT_INDIRECT=1 and run across 2 nodes to check that peers which are not
directly addressable are reported rather than translated.
"""

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
import rocshmem4py as rshmem

from iris.experimental.rocshmem_provider import RocshmemProvider

BLOCK_SIZE = 1024


@triton.jit
def _broadcast_kernel(
    data,
    results,
    peer_bases,
    n_elements,
    cur_rank,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Push this rank's values into `results` on every rank.

    Deliberately identical in shape to tests/unittests/test_store_triton.py:
    the whole point is that this is ordinary Iris device code, unaware that
    `peer_bases` came from rocSHMEM rather than an Iris heap.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(data + offsets, mask=mask)
    for dst_rank in range(num_ranks):
        iris.store(results + offsets, value, cur_rank, dst_rank, peer_bases, mask=mask)


def main():
    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rshmem.init_rocshmem_by_uniqueid(dist.group.WORLD)

    provider = RocshmemProvider()
    me, ws = provider.get_rank(), provider.get_num_ranks()
    assert ws >= 2, "need at least 2 ranks"

    # Two allocations from a non-Iris allocator.
    data, data_bases = provider.allocate_symmetric(BLOCK_SIZE, dtype=torch.float32)
    results, peer_bases = provider.allocate_symmetric(BLOCK_SIZE, dtype=torch.float32)

    amap = provider.symmetric_address_map(results)
    print(f"[rank{me}] direct={amap.direct} base={amap.allocation_base:#x} "
          f"bases={[hex(int(b)) for b in peer_bases.tolist()]}", flush=True)

    # A non-direct peer's base is 0, which would translate to a wild pointer
    # rather than error, so refuse instead. EXPECT_INDIRECT=1 tests that path.
    if os.environ.get("EXPECT_INDIRECT") == "1":
        detected = not amap.all_direct()
        print(f"[rank{me}] EXPECT_INDIRECT: indirect peers={amap.indirect_peers()} "
              f"detected={detected}", flush=True)
        res = [None] * ws
        dist.all_gather_object(res, detected)
        if me == 0:
            print("ROCSHMEM_PROVIDER_INDIRECT_RESULT:",
                  "PASS" if all(res) else "FAIL", flush=True)
        provider.barrier()
        provider.free(data)
        provider.free(results)
        dist.destroy_process_group()
        return 0

    assert amap.all_direct(), (
        f"[rank{me}] peers {amap.indirect_peers()} are not directly addressable; "
        "this prototype is IPC-only -- run all ranks on one node")

    # The invariant the device code actually depends on.
    assert int(peer_bases[me].item()) == results.data_ptr()

    # Rank 0 broadcasts its values; every rank should end up with them.
    data.fill_(float(me + 1))
    results.fill_(-1.0)
    torch.cuda.synchronize()
    provider.barrier()

    if me == 0:
        _broadcast_kernel[(1,)](
            data, results, peer_bases, BLOCK_SIZE, me,
            num_ranks=ws, BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
        )
        torch.cuda.synchronize()
    provider.barrier()

    want = 1.0  # rank 0's fill value
    ok = bool(torch.allclose(results, torch.full_like(results, want)))
    got = torch.unique(results)[:4].tolist()
    print(f"[rank{me}] results want={want} got={got} match={ok}", flush=True)

    res = [None] * ws
    dist.all_gather_object(res, ok)
    if me == 0:
        print("ROCSHMEM_PROVIDER_RESULT:", "PASS" if all(res) else "FAIL", flush=True)

    # Translate pointers in `results` using the table built from `data`: one
    # table should be valid for every allocation.
    provider.barrier()
    results.fill_(-1.0)
    torch.cuda.synchronize()
    provider.barrier()

    if me == 0:
        _broadcast_kernel[(1,)](
            data, results, data_bases, BLOCK_SIZE, me,   # data's table, results' pointers
            num_ranks=ws, BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
        )
        torch.cuda.synchronize()
    provider.barrier()

    xok = bool(torch.allclose(results, torch.full_like(results, want)))
    xgot = torch.unique(results)[:4].tolist()
    print(f"[rank{me}] cross-alloc want={want} got={xgot} match={xok}", flush=True)

    xres = [None] * ws
    dist.all_gather_object(xres, xok)
    if me == 0:
        print("ROCSHMEM_CROSS_ALLOC_RESULT:", "PASS" if all(xres) else "FAIL", flush=True)

    provider.barrier()
    provider.free(data)
    provider.free(results)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
