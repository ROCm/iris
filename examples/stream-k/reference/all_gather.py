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

    A_full = torch.randn(m, k, device=f"cuda:{rank}")
    B_full = torch.randn(k, n, device=f"cuda:{rank}")

    # Split B column-wise
    cols_per_gpu = n // world_size
    start_col = rank * cols_per_gpu
    end_col = start_col + cols_per_gpu
    B_local = B_full[:, start_col:end_col]

    # Allocate tensor for gathered results
    C_global = torch.empty(m, n, device=f"cuda:{rank}")


    def run_experiment():
        C_partial = A_full @ B_local
        gathered_parts = [C_global[:, i * cols_per_gpu: (i + 1) * cols_per_gpu] for i in range(world_size)]
        dist.all_gather(gathered_parts, C_partial)


    run_experiment()

    # Validation
    C_full = A_full @ B_full
    valid = torch.allclose(C_global, C_full, atol=1e-3)
    if valid:
        print(f"Rank {rank}: Validation passed! Distributed GEMM matches full GEMM.")
    else:
        print(f"Rank {rank}: Validation failed! Results do not match.")


    if benchmark:
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        ms = triton.testing.do_bench(run_experiment)
        print(f"Rank {rank}: {ms:.3f} ms  {perf(ms):.3f} tflops")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()