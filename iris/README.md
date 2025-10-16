<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Iris library

Python- and Triton-based library facilitating RDMAs for intra-node communication via IPC conduit.

Iris provides both a standard API using Triton's JIT functions and an experimental Gluon-based API using `@aggregate` and `@gluon.jit` for improved ergonomics.