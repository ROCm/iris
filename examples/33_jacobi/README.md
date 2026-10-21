<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Multi-GPU Jacobi iteration

This example solves a two-dimensional Jacobi stencil on multiple AMD GPUs. `nx` is the number of columns and `ny` is the number of rows. Each rank owns a contiguous strip of interior rows and keeps one halo row on either side. After every update, an Iris remote store sends the edge rows to neighboring ranks. An Iris all-reduce combines the residual so every rank makes the same convergence decision.

The left, right, top, and bottom boundaries are fixed at 100, 0, 50, and 0. The values are defined at the top of `example.py`. The split also handles grids whose interior rows do not divide evenly among GPUs.

## Run

From the repository root, start one process per GPU:

```terminal
torchrun --standalone --nproc_per_node=2 examples/33_jacobi/example.py --nx 512 --ny 512 --validate
```

Use `--nproc_per_node=4` or `8` to run on more GPUs. The number of GPUs cannot exceed `ny - 2`, the number of interior rows. `--max_iterations` defaults to 1000, `--tolerance` to `1e-6`, and `--heap_size` to 1 GiB per rank. The `--validate` option gathers the distributed grid and compares it with a single-GPU PyTorch reference after the same number of iterations.

## Test

```terminal
python tests/run_tests_distributed.py tests/examples/test_jacobi.py --num_ranks 2 -v
```

The test uses an uneven row split and checks the result against the PyTorch reference. CI has exercised this path with 2, 4, and 8 ranks.

![Jacobi iteration across AMD GPUs](https://github.com/user-attachments/assets/cf125da1-1942-476c-8406-578bc84ef62c)
