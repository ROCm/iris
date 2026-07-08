#!/usr/bin/env python3
"""Profile matmul_all_reduce_copy_engine to see GEMM vs collective timing.

Run with: ./benchmark/ops/profile_copy_engine.py
For one-shot: ./benchmark/ops/profile_copy_engine.py --axis_variant=one_shot
Then open the printed /tmp/copy_engine_<variant>_trace_rank0.json in chrome://tracing.
"""

import torch
import iris.bench as bench
import triton
import triton.language as tl
from iris.ops.matmul_all_reduce_copy_engine import (
    matmul_all_reduce_copy_engine as copy_engine,
    matmul_all_reduce_copy_engine_preamble,
)
from iris.ops import FusedConfig
from tritonblas.matmul import _make_matmul_selector

# Single shape for profiling
# M, N, K = 16384, 7168, 7168
M, N, K = 16384, 16384, 53248
dtype = torch.float16


@triton.jit()
def _stream_add_same_traffic_kernel(
    C,
    local_aux_buffer,
    remote_inbox,
    total_elements,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Force the same bytes as one-shot local reduce: 8 fp16 loads + 1 fp16 store."""
    pid = tl.program_id(0)
    total_blocks = total_elements // BLOCK_SIZE
    block_offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = tl.max_contiguous(tl.multiple_of(block_offsets, BLOCK_SIZE), BLOCK_SIZE)

    for block_id in tl.range(pid, total_blocks, NUM_SMS, num_stages=PIPELINE_STAGES):
        linear_base = block_id * BLOCK_SIZE
        linear_offsets = linear_base + block_offsets

        local_ptr = local_aux_buffer + linear_offsets
        local_ptr = tl.max_contiguous(tl.multiple_of(local_ptr, BLOCK_SIZE), BLOCK_SIZE)
        data0 = tl.load(local_ptr)

        if world_size == 8:
            src_offsets1 = 1 * total_elements + linear_offsets
            src_offsets2 = 2 * total_elements + linear_offsets
            src_offsets3 = 3 * total_elements + linear_offsets
            src_offsets4 = 4 * total_elements + linear_offsets
            src_offsets5 = 5 * total_elements + linear_offsets
            src_offsets6 = 6 * total_elements + linear_offsets
            src_offsets7 = 7 * total_elements + linear_offsets

            src_ptr1 = remote_inbox + src_offsets1
            src_ptr2 = remote_inbox + src_offsets2
            src_ptr3 = remote_inbox + src_offsets3
            src_ptr4 = remote_inbox + src_offsets4
            src_ptr5 = remote_inbox + src_offsets5
            src_ptr6 = remote_inbox + src_offsets6
            src_ptr7 = remote_inbox + src_offsets7
            src_ptr1 = tl.max_contiguous(tl.multiple_of(src_ptr1, BLOCK_SIZE), BLOCK_SIZE)
            src_ptr2 = tl.max_contiguous(tl.multiple_of(src_ptr2, BLOCK_SIZE), BLOCK_SIZE)
            src_ptr3 = tl.max_contiguous(tl.multiple_of(src_ptr3, BLOCK_SIZE), BLOCK_SIZE)
            src_ptr4 = tl.max_contiguous(tl.multiple_of(src_ptr4, BLOCK_SIZE), BLOCK_SIZE)
            src_ptr5 = tl.max_contiguous(tl.multiple_of(src_ptr5, BLOCK_SIZE), BLOCK_SIZE)
            src_ptr6 = tl.max_contiguous(tl.multiple_of(src_ptr6, BLOCK_SIZE), BLOCK_SIZE)
            src_ptr7 = tl.max_contiguous(tl.multiple_of(src_ptr7, BLOCK_SIZE), BLOCK_SIZE)

            data1 = tl.load(src_ptr1)
            data2 = tl.load(src_ptr2)
            data3 = tl.load(src_ptr3)
            data4 = tl.load(src_ptr4)
            data5 = tl.load(src_ptr5)
            data6 = tl.load(src_ptr6)
            data7 = tl.load(src_ptr7)
            data = ((data0 + data1) + (data2 + data3)) + ((data4 + data5) + (data6 + data7))
        else:
            data = data0
            for src_rank in tl.static_range(1, world_size):
                src_offsets = src_rank * total_elements + linear_offsets
                src_ptr = remote_inbox + src_offsets
                src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
                data += tl.load(src_ptr)

        out_ptr = C + linear_offsets
        out_ptr = tl.max_contiguous(tl.multiple_of(out_ptr, BLOCK_SIZE), BLOCK_SIZE)
        tl.store(out_ptr, data)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("variant", ["two_shot"])
def profile_matmul_all_reduce_copy_engine(state, ctx):
    """Profile copy_engine with PyTorch profiler."""
    rank = ctx.get_rank()
    variant = state["variant"]

    # Create tensors
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    selector = _make_matmul_selector(M, N, K, dtype, dtype, dtype, device, streamk=False)

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    # Create workspace
    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_copy_engine_preamble(ctx, C, A, B, config=config, selector=selector)

    flag_iteration = [0]

    def run_once(*, async_op: bool = False):
        copy_engine(
            ctx,
            C,
            A,
            B,
            async_op=async_op,
            config=config,
            workspace=workspace,
            flag_iteration=flag_iteration[0],
            copy_engine_transfers_preposted=False,
            split_completion_wait=(variant == "one_shot"),
        )
        flag_iteration[0] += 1

    # Register with benchmark framework
    state.set_flops(2 * M * N * K)
    state.exec(lambda: run_once(async_op=False))

    # After benchmark completes, do profiling run
    if rank == 0:
        print(f"\n{'='*120}")
        print(f"Running {variant} copy-engine profiler for {M}x{N}x{K}...")
        print(f"{'='*120}\n")

    profile_iteration = flag_iteration[0]

    # Warmup
    for _ in range(5):
        copy_engine(
            ctx,
            C,
            A,
            B,
            async_op=True,
            config=config,
            workspace=workspace,
            flag_iteration=profile_iteration,
            copy_engine_transfers_preposted=False,
            split_completion_wait=(variant == "one_shot"),
        )
        profile_iteration += 1
    torch.cuda.synchronize()

    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(10):
            copy_engine(
                ctx,
                C,
                A,
                B,
                async_op=True,
                config=config,
                workspace=workspace,
                flag_iteration=profile_iteration,
                copy_engine_transfers_preposted=False,
                split_completion_wait=(variant == "one_shot"),
            )
            profile_iteration += 1
        torch.cuda.synchronize()

    if rank == 0:
        # Print table sorted by CUDA time
        print("\n" + "="*120)
        print(f"{variant} CUDA Kernel Timeline (sorted by CUDA time)")
        print("="*120)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        # Export chrome trace for visualization
        trace_path = f"/tmp/copy_engine_{variant}_trace_rank0.json"
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace exported to {trace_path}")
        print("  Open in chrome://tracing to see timeline")
        print(f"{'='*120}\n")

    if variant == "one_shot":
        stream_copy_block_size = 1024
        stream_copy_total_elements = M * N
        stream_copy_num_sms = workspace.launch_params["reduce_num_sms"]
        stream_copy_grid = (
            min(stream_copy_num_sms, stream_copy_total_elements // stream_copy_block_size),
        )
        stream_add_local_buffer = workspace.a_inbox[rank * M : (rank + 1) * M, :]

        def run_stream_add(*, pipeline_stages: int):
            _stream_add_same_traffic_kernel[stream_copy_grid](
                C,
                stream_add_local_buffer,
                workspace.a_inbox,
                stream_copy_total_elements,
                world_size=ctx.get_num_ranks(),
                BLOCK_SIZE=stream_copy_block_size,
                NUM_SMS=stream_copy_num_sms,
                PIPELINE_STAGES=pipeline_stages,
                num_warps=4,
                num_stages=pipeline_stages,
            )

        for pipeline_stages in (1, 4):
            for _ in range(5):
                run_stream_add(pipeline_stages=pipeline_stages)
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            ) as stream_prof:
                for _ in range(10):
                    run_stream_add(pipeline_stages=pipeline_stages)
                torch.cuda.synchronize()

            if rank == 0:
                bytes_per_iter = stream_copy_total_elements * C.element_size() * (ctx.get_num_ranks() + 1)
                total_gib = bytes_per_iter * 10 / (1024 ** 3)
                print("\n" + "="*120)
                print(
                    f"one_shot stream-add same-traffic profile "
                    f"(num_stages={pipeline_stages}, {total_gib:.2f} GiB over 10 launches)"
                )
                print("="*120)
                print(stream_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                trace_path = f"/tmp/stream_add_same_traffic_stages{pipeline_stages}_trace_rank0.json"
                stream_prof.export_chrome_trace(trace_path)
                print(f"\nChrome trace exported to {trace_path}")
                print(f"{'='*120}\n")


if __name__ == "__main__":
    bench.main()
