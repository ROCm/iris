
import statistics
import math

def get_empty_cache_for_benchmark():
    import torch
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

def clear_cache(cache):
    cache.zero_()
      
def create_timing_event():
    import torch
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

