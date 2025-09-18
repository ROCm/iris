Of course. Here is the updated README with the new "Simple Example Run" section and the corrected file paths.

-----

# Fused All-Gather + GEMM

This project provides an example of a distributed All-Gather + GEMM kernel, a fundamental building block in many large AI models. It explores two distinct architectural patterns for fusing communication and computation: a **Pull model** and a **Push model**.

These novel implementations are designed to hide communication latency and reduce the overheads associated with standard, non-fused library calls. The core kernel implementations are located in `examples/14_all_gather_gemm/`.

Comparisons are performed against a baseline using the RCCL All-Gather collective and `torch.matmul`.

-----

## Architectural Patterns: Pull vs. Push

The two main patterns explored are:

### 1\. Pull Model

In the **Pull model**, the consumer (GEMM kernel) takes full control. It actively "pulls" data from remote GPUs as it is needed using an `iris.load` instruction. The communication is fused directly into a single, persistent compute kernel.

### 2\. Push Model

The **Push model** decouples communication and computation. A dedicated producer kernel "pushes" data to a remote inbox using `iris.store`, and the consumer (GEMM kernel) waits for a synchronization signal before performing a fast local load from that inbox.

-----

## Usage

### Simple Example Run

To run a minimal, standalone example that demonstrates the kernel's functionality and validates its output for a single configuration, use the `example_run` scripts.

**Pull Model:**

```terminal
python examples/14_all_gather_gemm/example_run_pull.py --num_ranks 8
```

**Push Model:**

```terminal
python examples/14_all_gather_gemm/example_run_push.py --num_ranks 8
```

### Validation and Benchmarking

For more comprehensive testing, dedicated scripts in the `benchmark/examples/` directory handle both correctness validation and performance benchmarking across a range of configurations. The behavior of these scripts is controlled by flags.

The scripts run a sweep of configurations defined in the JSON file at `dataset/ag_gemm.json`.

#### Validation (-v)

To verify the numerical correctness of an implementation against a PyTorch reference, run its benchmark script with the `-v` or `--validate` flag.

**Pull Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_pull.py --num_ranks 8 --validate
```

**Push Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_push.py --num_ranks 8 --validate
```

#### Benchmarking (-b)

To run the full performance benchmark sweep and save the results as `.json` files into the `results/` directory, use the `-b` or `--benchmark` flag.

**Pull Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_pull.py --num_ranks 8 --benchmark
```

**Push Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_push.py --num_ranks 8 --benchmark
```