<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Iris library

Python- and Triton-based library facilitating RDMAs for intra-node communication via IPC conduit.

The `finegrained_alloc` directory contains a C library interface for fine-grained allocation. The plugin is required to redirect PyTorch allocation to fine-grained memory.