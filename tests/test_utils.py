# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test utilities for PyTorch distributed testing.
"""

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes
import pytest
from typing import Callable, Any, Dict
import os


def run_distributed_test(test_func: Callable, num_ranks: int = 2, **kwargs) -> Any:
    """
    Run a test function in a distributed environment.
    
    Args:
        test_func: Test function to run. Should accept (local_rank, world_size, **kwargs)
        num_ranks: Number of processes to spawn
        **kwargs: Additional arguments to pass to test_func
        
    Returns:
        Result from rank 0, or None if test_func doesn't return anything
    """
    results = {}
    
    def _worker(local_rank: int, world_size: int, init_url: str, test_func: Callable, kwargs: Dict):
        backend = "gloo"  # Use gloo backend for CPU testing
        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_url,
                world_size=world_size,
                rank=local_rank
            )
            
            result = test_func(local_rank, world_size, **kwargs)
            if local_rank == 0:
                results[0] = result
        finally:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
    
    init_url = f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}"  # Avoid port conflicts
    start_processes(
        fn=_worker,
        args=(num_ranks, init_url, test_func, kwargs),
        nprocs=num_ranks,
        join=True,
    )
    
    return results.get(0)


def distributed_test(num_ranks: int = 2):
    """
    Decorator to mark a test as distributed.
    
    Usage:
        @distributed_test(num_ranks=4)
        def test_my_distributed_function(local_rank, world_size):
            # Test logic here
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            return run_distributed_test(test_func, num_ranks, *args, **kwargs)
        return wrapper
    return decorator


@pytest.fixture
def iris_distributed_context():
    """
    Pytest fixture for creating iris contexts in tests.
    
    This handles the distributed setup automatically for simple unit tests.
    """
    def create_iris_context(heap_size: int = 1 << 20, num_ranks: int = 2):
        """Create an iris context with distributed setup."""
        result = {}
        
        def _worker(local_rank: int, world_size: int, init_url: str, heap_size: int):
            backend = "gloo"
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_url,
                    world_size=world_size,
                    rank=local_rank
                )
                
                # Mock HIP functions for testing
                import iris
                ctx = iris.iris(heap_size)
                if local_rank == 0:
                    result[0] = ctx
            except Exception as e:
                if local_rank == 0:
                    result[0] = e
            finally:
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
        
        init_url = f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}"
        start_processes(
            fn=_worker,
            args=(num_ranks, init_url, heap_size),
            nprocs=num_ranks,
            join=True,
        )
        
        ctx = result.get(0)
        if isinstance(ctx, Exception):
            raise ctx
        return ctx
    
    return create_iris_context