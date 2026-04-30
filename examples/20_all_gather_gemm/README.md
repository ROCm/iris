# Fused All-Gather + GEMM

This folder provides an example of a distributed All-Gather + GEMM kernel. It explores two distinct patterns for fusing communication and computation: a **Pull model** and a **Push model**.

The core kernel implementations are located in `examples/14_all_gather_gemm/`.

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
python benchmark/examples/benchmark_all_gather_gemm_pull.py --num_ranks 8 -v
```

**Push Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_push.py --num_ranks 8 -v
```

#### Benchmarking (-b)

To run the full performance benchmark sweep and save the results as `.json` files into the `results/` directory, use the `-b` or `--benchmark` flag.

**Pull Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_pull.py --num_ranks 8 -b
```

**Push Model:**

```terminal
python benchmark/examples/benchmark_all_gather_gemm_push.py --num_ranks 8 -b
```

#### RCCL + Torch

To validate and benchmark the RCCL + `torch.matmul` implementation, follow the same steps as the pull/push versions.

```terminal
python examples/benchmark/reference/all_gather_gemm/benchmark_rccl_torch.py --num_ranks 8 -b
```