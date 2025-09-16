<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Fused Flash Decode Attention

This is an example for a distributed Flash Decode kernel designed to accelerate LLM Inference. Part of the code is adapted from [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed).

This is a novel implementation that fuses communication and computation, diminshing the collective kernel launch latencies and the associated waits.

The core layer implementation is in `examples/13_flash_decode/flash_decode_fused_layer.py` while the Triton fused kernels are defined in `examples/13_flash_decode/decode_kernels.py`. 

We perform comparisons against the RCCL baseline.

---

## Usage

### Simple Example

To simply do a test run of the code, run:
```terminal
python examples/13_flash_decode/example_run.py
```
This example will run by default on 8 GPUs. Use the `--num_ranks` flag to select the number of GPUs.

### Validation

These scripts use `pytest` to verify the numerical correctness of each implementation against a standard PyTorch reference.

**Iris**

```terminal
python tests/run_tests_distributed.py tests/examples/test_flash_decode.py --num_ranks 8
```

**RCCL**

```terminal
python tests/run_tests_distributed.py examples/benchmark/reference/flash_decode_rccl/validate_flash_decode_rccl.py --num_ranks 8
```

### Benchmarking

These scripts run a sweep of configurations and save performance results as `.json` files into the `results/` directory.

**Iris**

```terminal
python benchmark/examples/benchmark_flash_decode.py --num_ranks 8
```

**RCCL**

```terminal
torchrun --nproc_per_node=8 examples/benchmark/reference/flash_decode_rccl/benchmark_flash_decode_rccl.py
```


