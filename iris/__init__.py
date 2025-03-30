# __init__.py

import os
import torch
import subprocess

# Pipe allocations via finegrained allocator
current_dir = os.path.dirname(__file__)
finegrained_alloc_path = os.path.join(
    current_dir, "finegrained_alloc", "libfinegrained_allocator.so"
)

def compile():
    if os.path.exists(finegrained_alloc_path):
        return

    print("Fine-grained allocator shared library not found. Building...")

    name = "finegrained_allocator"
    src_file = os.path.join(current_dir, "finegrained_alloc", f"{name}.hip")

    basic_warnings = [
        "-Wall", "-Wextra", "-Werror"
    ]
    strict_warnings = [
        "-pedantic", "-Wshadow", "-Wnon-virtual-dtor", "-Wold-style-cast",
        "-Wcast-align", "-Woverloaded-virtual", "-Wconversion",
        "-Wsign-conversion", "-Wnull-dereference", "-Wdouble-promotion", "-Wformat=2"
    ]
    std_flags = ["-std=c++17"]
    output_flags = ["-shared", "-fPIC", "-o", finegrained_alloc_path]

    cmd = ["hipcc"] + basic_warnings + strict_warnings + std_flags + output_flags + [src_file]

    try:
        subprocess.run(cmd, cwd=os.path.dirname(src_file), check=True)
        print(f"Built: {finegrained_alloc_path}")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with return code {e.returncode}")
        assert False, "hipcc build failed"

compile()
    
finegrained_allocator = torch.cuda.memory.CUDAPluggableAllocator(
    finegrained_alloc_path,
    "finegrained_hipMalloc",
    "finegrained_hipFree",
)
torch.cuda.memory.change_current_allocator(finegrained_allocator)


from .iris import (
    Iris,
    translate,
    get,
    put,
    atomic_add,
    atomic_sub,
    atomic_cas,
    atomic_xchg,
)
from .util import (
    do_bench
)

__all__ = [
    "Iris",
    "translate",
    "get",
    "put",
    "atomic_add",
    "atomic_sub",
    "atomic_cas",
    "atomic_xchg,",
    "do_bench,",
]
