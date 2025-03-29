import triton
import triton.language as tl

import json
import numpy as np
import os
import torch
import statistics
import math


# Communication Algorithms
ALL_SCATTER = tl.constexpr(1)
ALL_REDUCE = tl.constexpr(2)
ONE_SHOT = tl.constexpr(3)
NONE = tl.constexpr(4)




class JSONWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)

    def add_field(self, key, value):
        self.data[key] = value

    def _write_to_file(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)
    def display(self):
        print(json.dumps(self.data, indent=4))
    def flush(self):
        self._write_to_file()


class Timestamps:
    def __init__(self, num_tiles):
        self.max_ts = torch.iinfo(torch.int64).max
        self.min_ts = 0
        self.mm_begin_timestamp = torch.empty(num_tiles, dtype=torch.int64, device='cuda')
        self.mm_end_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device='cuda')

        self.comm_begin_timestamp = torch.empty(num_tiles, dtype=torch.int64, device='cuda')
        self.comm_middle_min_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device='cuda')
        self.comm_middle_max_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device='cuda')
        self.comm_end_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device='cuda')

    def reset(self):
        self.mm_begin_timestamp.fill_(self.max_ts)
        self.mm_end_timestamp.fill_(self.min_ts)
        
        self.comm_begin_timestamp.fill_(self.max_ts)
        self.comm_middle_min_timestamp.fill_(self.max_ts)
        self.comm_middle_max_timestamp.fill_(self.min_ts)
        self.comm_end_timestamp.fill_(self.min_ts)
        
    def to_json(self, filename, gpu_freq):
        cycles_to_us = lambda cycles: (cycles / gpu_freq)

        gemm_begin_us = cycles_to_us(self.mm_begin_timestamp.cpu().numpy())
        gemm_end_us = cycles_to_us(self.mm_end_timestamp.cpu().numpy())

        comm_begin_us = cycles_to_us(self.comm_begin_timestamp.cpu().numpy())
        poll_end_us = cycles_to_us(self.comm_middle_max_timestamp.cpu().numpy())
        op_begin_us = cycles_to_us(self.comm_middle_min_timestamp.cpu().numpy())
        op_end_us = cycles_to_us(self.comm_end_timestamp.cpu().numpy())


        min_timestamp = min(np.min(gemm_begin_us),
                            np.min(gemm_end_us),
                            np.min(comm_begin_us),
                            np.min(poll_end_us),
                            np.min(op_begin_us),
                            np.min(op_end_us))

        gemm_begin_us = gemm_begin_us - min_timestamp
        gemm_end_us = gemm_end_us - min_timestamp
        comm_begin_us = comm_begin_us - min_timestamp
        poll_end_us = poll_end_us - min_timestamp
        op_begin_us = op_begin_us - min_timestamp
        op_end_us = op_end_us - min_timestamp

        data = [
            {"tile_id": i,
            "gemm_begin": int(gemm_begin),
            "gemm_end":   int(gemm_end),
            "poll_begin": int(comm_begin),
            "poll_end":   int(poll_end),
            "op_begin":   int(op_begin),
            "op_end":     int(op_end,),
            "comm_begin": int(comm_begin),
            "comm_end":   int(op_end,)}
            for i, (gemm_begin, gemm_end, comm_begin,
                    poll_end, op_begin, op_end) in enumerate(zip(gemm_begin_us,
                                        gemm_end_us,
                                        comm_begin_us,
                                        poll_end_us,
                                        op_begin_us,
                                        op_end_us))
        ]
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

def get_empty_cache_for_benchmark():
    import torch
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

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
                             
def do_bench(fn, barrier_fn, n_warmup=25, n_repeat=100, quantiles=None, return_mode="mean"):

    # Wait for anything that happened before
    barrier_fn()
    fn()
    barrier_fn()
    # Wait for all GPUs to finish their work

    cache = get_empty_cache_for_benchmark()

    start_event = [create_timing_event() for i in range(n_repeat)]
    end_event = [create_timing_event() for i in range(n_repeat)]
    
    # Warm-up
    for _ in range(n_warmup):
        barrier_fn() # Wait for all GPUs before we clear the cache
        clear_cache(cache)
        barrier_fn() # Wait for clearing the cache before launching any kernels
        fn()
        
    # Benchmark
    for i in range(n_repeat):
        barrier_fn() # Wait for all GPUs before we clear the cache
        clear_cache(cache)
        barrier_fn() # Wait for clearing the cache before launching any kernels
        start_event[i].record()
        fn()
        end_event[i].record()
    
    barrier_fn() # Record clocks barrier

    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def is_triton_interpret_set():
    return "TRITON_INTERPRET" in os.environ

@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="""s_waitcnt vmcnt(0)
        s_memrealtime $0
        s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1
    )
    return tmp


