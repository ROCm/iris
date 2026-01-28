# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import statistics
import math
import torch


def get_empty_cache_for_benchmark():
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device="cuda")


def clear_cache(cache):
    cache.zero_()


def create_timing_event():
    return torch.cuda.Event(enable_timing=True)


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def do_bench(
    fn,
    barrier_fn=lambda: None,
    preamble_fn=lambda: None,
    n_warmup=25,
    n_repeat=100,
    quantiles=None,
    return_mode="mean",
):
    """
    Benchmark a function by timing its execution.

    Args:
        fn (callable): Function to benchmark.
        barrier_fn (callable, optional): Function to call for synchronization. Default: no-op.
        preamble_fn (callable, optional): Function to call before each execution. Default: no-op.
        n_warmup (int, optional): Number of warmup iterations. Default: 25.
        n_repeat (int, optional): Number of timing iterations. Default: 100.
        quantiles (list, optional): Quantiles to return instead of summary statistic. Default: None.
        return_mode (str, optional): Summary statistic to return ("mean", "min", "max", "median", "all"). Default: "mean".

    Returns:
        float or list: Timing result(s) in milliseconds.

    Example:
        >>> import iris
        >>> iris_ctx = iris.iris(1 << 20)
        >>> def test_fn():
        >>>     tensor = iris_ctx.zeros(1000, 1000)
        >>> time_ms = iris.do_bench(test_fn, barrier_fn=iris_ctx.barrier)
    """
    # Wait for anything that happened before
    barrier_fn()
    preamble_fn()
    fn()
    barrier_fn()
    # Wait for all GPUs to finish their work

    cache = get_empty_cache_for_benchmark()

    start_event = [create_timing_event() for i in range(n_repeat)]
    end_event = [create_timing_event() for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        barrier_fn()  # Wait for all GPUs before we clear the cache
        preamble_fn()
        clear_cache(cache)
        barrier_fn()  # Wait for clearing the cache before launching any kernels
        fn()

    # Benchmark
    for i in range(n_repeat):
        barrier_fn()  # Wait for all GPUs before we clear the cache
        preamble_fn()
        clear_cache(cache)
        barrier_fn()  # Wait for clearing the cache before launching any kernels
        start_event[i].record()
        fn()
        end_event[i].record()

    barrier_fn()  # Record clocks barrier

    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)
