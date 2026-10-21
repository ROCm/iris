#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import gc
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import iris


# Load the real example file so this test follows the same code CI will run.
root = Path(__file__).resolve()
while not (root / "tests").is_dir() or not (root / "examples").is_dir():
    if root == root.parent:
        raise FileNotFoundError("Could not find project root")
    root = root.parent

path = root / "examples" / "33_jacobi" / "example.py"
spec = importlib.util.spec_from_file_location("jacobi_example", path)
jac = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(jac)


# This runs the real distributed path on a small uneven grid.
# The reference comparison catches wrong row splits or halo writes.
def test_multi_gpu_jacobi_matches_reference():
    if not dist.is_initialized():
        pytest.skip("torch.distributed is not initialized")

    nranks = dist.get_world_size()
    if nranks < 2:
        pytest.skip("Jacobi halo exchange requires at least two ranks")

    ctx = None
    try:
        ctx = iris.iris(heap_size=1 << 26)
        rank = ctx.get_rank()
        nx = 64
        ny = 32 * nranks + 1
        nit = 8

        _, _, owned_rows = jac.split_rows(ny, rank, nranks)
        max_owned_rows = (ny - 2 + nranks - 1) // nranks
        buffer_rows = max_owned_rows + 2
        upper_rank = rank - 1 if rank > 0 else -1
        lower_rank = rank + 1 if rank < nranks - 1 else -1
        upper_owned_rows = jac.split_rows(ny, upper_rank, nranks)[2] if upper_rank >= 0 else 0

        cur = ctx.zeros((buffer_rows, nx), dtype=torch.float32)
        cur[:, 0] = jac.LEFT_BOUNDARY
        cur[:, -1] = jac.RIGHT_BOUNDARY
        if rank == 0:
            cur[0] = jac.TOP_BOUNDARY
        if rank == nranks - 1:
            cur[owned_rows + 1] = jac.BOTTOM_BOUNDARY

        nxt = ctx.zeros((buffer_rows, nx), dtype=torch.float32)
        nxt.copy_(cur)
        dev_ctx = ctx.get_device_context()
        err = ctx.zeros((1, 1), dtype=torch.float32)
        all_err = ctx.zeros((1, 1), dtype=torch.float32)

        # Seed halos first then run the same real step function as main.
        jac.push_halos(ctx, dev_ctx, cur, nx, owned_rows, upper_rank, lower_rank, upper_owned_rows, rank, nranks)
        for _ in range(nit):
            jac.do_step(
                ctx,
                dev_ctx,
                cur,
                nxt,
                err,
                all_err,
                nx,
                owned_rows,
                upper_rank,
                lower_rank,
                upper_owned_rows,
                rank,
                nranks,
            )
            cur, nxt = nxt, cur

        full = jac.gather_grid(ctx, cur, nx, ny, owned_rows, max_owned_rows, nranks)
        ref = jac.ref_jacobi(nx, ny, nit, full.device)
        torch.testing.assert_close(full, ref, atol=1e-3, rtol=1e-4)

    finally:
        # Another rank may already have failed so cleanup barrier is best effort.
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()
