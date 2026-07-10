#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl barrier collective."""

import torch
import iris.bench as bench
from iris.ccl import Config


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
def barrier(state, ctx):
    state.set_bytes(0)

    state.exec(
        lambda: ctx.ccl.barrier(async_op=True),
    )


if __name__ == "__main__":
    bench.main()
