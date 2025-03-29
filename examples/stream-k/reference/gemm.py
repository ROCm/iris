import torch
import triton

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    m, n, k = 4864, 4096, 8256
    benchmark = True

    rank = 0
    A_full = torch.randn(m, k, device=f"cuda:{rank}")
    B_full = torch.randn(k, n, device=f"cuda:{rank}")

    def run_experiment():
        return A_full @ B_full

    C_global = run_experiment()

    if benchmark:
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        ms = triton.testing.do_bench(run_experiment)
        print(f"Rank {rank}: {ms:.3f} ms  {perf(ms):.3f} TFLOPS")

if __name__ == "__main__":
    main()