#!/usr/bin/env python3
import os
import sys
import argparse
import shutil

# Example:
# cd examples/29_ops_all_gather_matmul
# torchrun --nproc_per_node=4 --standalone ../../scripts/roccap_wrapper.py -k _fused_all_gather_matmul_kernel example.py -m 1024 -n 128

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kernel", type=str, default="_fused_all_gather_matmul_kernel")
parser.add_argument("--skip-roccap", action="store_true", help="Skip roccap and run script directly")

# Everything else: first is the script to run, rest are passed through to it
parsed, unknown = parser.parse_known_args()
if not unknown:
    sys.exit("Usage: roccap_wrapper.py [-k KERNEL] [--skip-roccap] <script> [script_args...]")
child_script = unknown[0]
child_args = unknown[1:]

# Get rank from torchrun environment
rank = os.environ.get("RANK", "0")

# Hardcoded dispatch filter: only capture dispatches
DISP_FILTER = f"{parsed.kernel}/0-"

print(f"sys.executable: {sys.executable}")
# Build roccap command: capture --loglevel trace --file <output> --disp <filter> python3 <script> [args...]
roccap_args = [
    "capture",
    "--loglevel",
    "trace",
    "--file",
    f"{parsed.kernel}_rank_{rank}.cap",
    "--disp",
    DISP_FILTER,
    sys.executable,
    child_script,
] + child_args

# Set simulation env so Iris uses torch allocator
os.environ["IRIS_SIMULATION"] = "1"
# Pass kernel name so Iris can name heap_bases output to match -k (e.g. persistent_all_gather_heap_bases.json)
os.environ["IRIS_HEAP_BASES_PREFIX"] = parsed.kernel
# Disable PyTorch caching allocator for simple allocations in simulation
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

if parsed.skip_roccap:
    # Skip roccap and run script directly
    # Resolve child_script path relative to current working directory
    if not os.path.isabs(child_script):
        child_script_abs = os.path.abspath(child_script)
        if os.path.exists(child_script_abs):
            child_script = child_script_abs
        elif os.path.exists(child_script):
            # Keep relative path if it exists
            pass
        else:
            sys.exit(f"Error: Script not found: {child_script} (resolved: {child_script_abs})")
    print(f"Executing the command: {sys.executable} {child_script} {' '.join(child_args)}")
    os.execvp(sys.executable, [sys.executable, child_script] + child_args)
else:
    # Run through roccap
    roccap_path = shutil.which("roccap")
    if roccap_path is None:
        sys.exit("roccap not found in PATH. Install it or activate the environment that provides it.")
    print(f"Executing the command: roccap {' '.join(roccap_args)}")
    os.execvp(roccap_path, ["roccap"] + roccap_args)
