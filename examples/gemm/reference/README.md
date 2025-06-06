<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Reference implementation using RCCL

Implements two variants of the Distributed GEMM.

1. Split B column wise, then do all gather
2. Split B row-wise, then do all reduce.

See `run.sh` for how to run.