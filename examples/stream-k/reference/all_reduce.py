import torch
import torch.distributed as dist
import triton

def main():

    torch.manual_seed(42)

    m, n, k = 4864, 4096, 8256
    benchmark = True

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    A_full = torch.randn(m, k).cuda(rank)
    B_full = torch.randn(k, n).cuda(rank)
    rows_per_gpu = k // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    B_local = B_full[start_row:end_row, :]
    A_local = A_full[:, start_row:end_row]
    C_global = torch.zeros((m, n), device=f"cuda:{rank}")

    def run_experiment():
        global C_partial
        C_partial = A_local @ B_local
        dist.all_reduce(C_partial, op=dist.ReduceOp.SUM)


    run_experiment()

    C_global.copy_(C_partial)

    # Validation step
    C_full = A_full @ B_full
    valid = torch.allclose(C_global, C_full, atol=1e-2)
    if valid:
        print(f"Rank {rank}: Validation passed! Distributed GEMM matches full GEMM.")
    else:
        print(f"Rank {rank}: Validation failed! Results do not match.")


    if benchmark:
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        ms = triton.testing.do_bench(lambda: run_experiment())
        print(f"Rank {rank}: {ms:.3f} ms  {perf(ms):.3f} tflops")


    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()