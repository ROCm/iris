#!/usr/bin/env python3
"""
Benchmark flat-2D gluon all-gather: use a single 1D arange over BLOCK_M*BLOCK_N
elements, compute row/col via div/mod, and do one load + world_size stores per tile.
"""
import os
import torch
import torch.distributed as dist

M, N = 8192, 8192
DTYPE = torch.float16
HEAP_SIZE = 2**33
n_warmup = 10
n_repeat = 50


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    import triton.language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from iris.experimental.iris_gluon import IrisDeviceCtx
    import iris.experimental.iris_gluon as iris_gluon
    from iris.ccl.utils import extract_group_info

    # ---- Flat-2D kernel ----
    @gluon.jit
    def ag_flat2d(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr, output_ptr,
        M, N,
        stride_in_m, stride_in_n,
        stride_out_m, stride_out_n,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
    ):
        ctx = IrisDeviceCtx.initialize(context_tensor)
        pid = gl.program_id(0)

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        # Flat 1D layout covering BLOCK_SIZE_M * BLOCK_SIZE_N elements
        TOTAL_ELEMS: gl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N
        ELEMS_PER_THREAD: gl.constexpr = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)
        flat_layout: gl.constexpr = gl.BlockedLayout(
            [ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0]
        )

        # Pre-compute heap base for hoisted translation
        local_base = gl.load(ctx.heap_bases + iris_rank)

        for tile_id in range(pid, total_tiles, COMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Flat index -> 2D row/col
            flat_idx = gl.arange(0, TOTAL_ELEMS, layout=flat_layout)
            row_local = flat_idx // BLOCK_SIZE_N  # row within tile [0, BLOCK_SIZE_M)
            col_local = flat_idx % BLOCK_SIZE_N   # col within tile [0, BLOCK_SIZE_N)

            # Global row/col
            row = pid_m * BLOCK_SIZE_M + row_local
            col = pid_n * BLOCK_SIZE_N + col_local

            mask = (row < M) & (col < N)

            # Single flat load of the entire tile
            input_offsets = row * stride_in_m + col * stride_in_n
            data = gl.load(input_ptr + input_offsets, mask=mask, other=0.0)

            # Output: this rank's data goes to output[group_rank * M + row, col]
            output_row = group_rank * M + row
            output_offsets = output_row * stride_out_m + col * stride_out_n

            # Traffic-shaped stores to all ranks
            for rank_idx in range(world_size):
                dest_idx = (group_rank + rank_idx) % world_size
                target_iris_rank = rank_start + dest_idx * rank_stride
                output_ptrs = output_ptr + output_offsets

                if dest_idx == group_rank:
                    gl.store(output_ptrs, data, mask=mask, cache_modifier=".wt")
                else:
                    target_base = gl.load(ctx.heap_bases + target_iris_rank)
                    ptr_delta = target_base - local_base
                    output_ptrs_int = tl.cast(output_ptrs, gl.uint64)
                    remote_ptrs_int = output_ptrs_int + ptr_delta
                    remote_ptrs = tl.cast(remote_ptrs_int, output_ptrs.dtype)
                    gl.store(remote_ptrs, data, mask=mask)

    # ---- 1D hoisted for comparison ----
    @gluon.jit
    def ag_1d_hoisted(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr, output_ptr,
        M, N,
        stride_in_m, stride_in_n,
        stride_out_m, stride_out_n,
        group_rank: gl.constexpr,
        iris_rank: gl.constexpr,
        world_size: gl.constexpr,
        rank_start: gl.constexpr,
        rank_stride: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        COMM_SMS: gl.constexpr,
        THREADS_PER_WARP: gl.constexpr,
        WARPS_PER_CTA: gl.constexpr,
    ):
        ctx = IrisDeviceCtx.initialize(context_tensor)
        pid = gl.program_id(0)

        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        total_tiles = num_pid_m * num_pid_n

        ELEMS_PER_THREAD: gl.constexpr = BLOCK_SIZE_N // (THREADS_PER_WARP * WARPS_PER_CTA)
        col_layout: gl.constexpr = gl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

        local_base = gl.load(ctx.heap_bases + iris_rank)

        for tile_id in range(pid, total_tiles, COMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=col_layout)) % N
            rn = gl.max_contiguous(gl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            col_offsets_in = rn * stride_in_n
            col_offsets_out = rn * stride_out_n
            col_mask = rn < N
            rm_base = pid_m * BLOCK_SIZE_M

            for i in range(BLOCK_SIZE_M):
                row_idx = (rm_base + i) % M
                if row_idx < M:
                    input_addr = input_ptr + row_idx * stride_in_m + col_offsets_in
                    data = gl.load(input_addr, mask=col_mask)
                    output_offset = (group_rank * M + row_idx) * stride_out_m + col_offsets_out

                    for rank_idx in range(world_size):
                        dest_idx = (group_rank + rank_idx) % world_size
                        target_iris_rank = rank_start + dest_idx * rank_stride
                        output_addr = output_ptr + output_offset

                        if dest_idx == group_rank:
                            gl.store(output_addr, data, mask=col_mask, cache_modifier=".wt")
                        else:
                            target_base = gl.load(ctx.heap_bases + target_iris_rank)
                            ptr_delta = target_base - local_base
                            output_addr_int = tl.cast(output_addr, gl.uint64)
                            remote_addr_int = output_addr_int + ptr_delta
                            remote_addr = tl.cast(remote_addr_int, output_addr.dtype)
                            gl.store(remote_addr, data, mask=col_mask)

    # ---- Bench helper ----
    def bench_kernel(label, kernel_fn, shmem, inp, out, ctx_tensor,
                     ri, rg, ws, rs, rstride, num_cus, bsm, bsn):
        for _ in range(n_warmup):
            out.zero_()
            shmem.barrier()
            kernel_fn[(num_cus,)](
                IrisDeviceCtx, ctx_tensor, inp, out, M, N,
                inp.stride(0), inp.stride(1), out.stride(0), out.stride(1),
                ri, rg, ws, rs, rstride, bsm, bsn, 4, num_cus, 64, 4,
                num_warps=4,
            )
            shmem.barrier()

        # Validate
        out.zero_()
        inp.fill_(float(rank + 1))
        shmem.barrier()
        kernel_fn[(num_cus,)](
            IrisDeviceCtx, ctx_tensor, inp, out, M, N,
            inp.stride(0), inp.stride(1), out.stride(0), out.stride(1),
            ri, rg, ws, rs, rstride, bsm, bsn, 4, num_cus, 64, 4,
            num_warps=4,
        )
        shmem.barrier()
        expected = torch.zeros(ws * M, N, dtype=DTYPE, device=f"cuda:{rank}")
        for r in range(ws):
            expected[r * M : (r + 1) * M, :] = float(r + 1)
        valid = torch.allclose(out, expected, atol=1e-3)

        # Bench
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        shmem.barrier()
        start.record()
        for _ in range(n_repeat):
            kernel_fn[(num_cus,)](
                IrisDeviceCtx, ctx_tensor, inp, out, M, N,
                inp.stride(0), inp.stride(1), out.stride(0), out.stride(1),
                ri, rg, ws, rs, rstride, bsm, bsn, 4, num_cus, 64, 4,
                num_warps=4,
            )
        end.record()
        torch.cuda.synchronize()
        shmem.barrier()

        ms = start.elapsed_time(end) / n_repeat
        total_bytes = (ws - 1) * M * N * 2
        bw = (total_bytes / 1e9) / (ms / 1e3)
        per_link = (M * N * 2 / 1e9) / (ms / 1e3)
        status = "PASS" if valid else "FAIL"
        if rank == 0:
            print(f"  {label:<30s}  {ms:>8.3f} ms  {bw:>8.2f} GB/s  {per_link:>8.2f}/link  [{status}]")

    # ---- Main bench ----
    shmem = iris_gluon.iris(HEAP_SIZE)
    inp = shmem.zeros((M, N), dtype=DTYPE)
    out = shmem.zeros((world_size * M, N), dtype=DTYPE)
    inp.fill_(float(rank + 1))
    ctx_tensor = shmem.get_device_context()
    ri, rg, ws, rs, rstride = extract_group_info(None, shmem)

    if rank == 0:
        print(f"world_size={world_size}, M={M}, N={N}, fp16")

    for num_cus in [16, 32, 64, 96]:
        if rank == 0:
            print(f"\n--- {num_cus} CUs ---")

        # 1D hoisted (baseline)
        bench_kernel("1D hoisted 32x1024", ag_1d_hoisted, shmem, inp, out, ctx_tensor,
                     ri, rg, ws, rs, rstride, num_cus, 32, 1024)

        # Flat-2D with different tile sizes
        for bsm, bsn in [(8, 256), (4, 512), (8, 512), (16, 512), (32, 256)]:
            total = bsm * bsn
            ept = total // (64 * 4)
            if ept < 1:
                continue
            label = f"flat2D {bsm}x{bsn}"
            try:
                bench_kernel(label, ag_flat2d, shmem, inp, out, ctx_tensor,
                             ri, rg, ws, rs, rstride, num_cus, bsm, bsn)
            except Exception as e:
                if rank == 0:
                    err = str(e).split('\n')[0][:60]
                    print(f"  {label:<30s}  ERROR: {err}")

    del inp, out, shmem
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
