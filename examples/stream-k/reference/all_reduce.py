#!/usr/bin/env python3

import torch
import torch.distributed as dist
import random
import iris
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import *

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=4864, help="Number of rows in matrix A")
    parser.add_argument(
        "-n", type=int, default=4096, help="Number of columns in matrix B"
    )
    parser.add_argument(
        "-k", type=int, default=8256, help="Common dimension between matrices A and B"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Enable validation mode"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmarking mode"
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    return vars(parser.parse_args())

def main():

    args = parse_args()

    m = args["m"]
    n = args["n"]
    k = args["k"]
    validate = args["validate"]
    benchmark = args["benchmark"]

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    args["M"] = args["m"]
    args["N"] = args["n"]
    args["K"] = args["k"]
    
    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    
    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    A_full = torch.randn(m, k).cuda(rank)
    B_full = torch.randn(k, n).cuda(rank)
    rows_per_gpu = k // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    B_local = B_full[start_row:end_row, :]
    A_local = A_full[:, start_row:end_row]
    C_global = torch.zeros((m, n), device=f"cuda:{rank}")

    args["k"] = args["k"] // world_size


    for key, value in args.items():
        json_writer.add_field(key, value)
        
    def run_experiment():
        global C_partial
        C_partial = A_local @ B_local
        dist.all_reduce(C_partial, op=dist.ReduceOp.SUM)

    run_experiment()

    C_global.copy_(C_partial)

    # Validation step
    if validate:
        C_full = A_full @ B_full
        valid = torch.allclose(C_global, C_full, atol=1)
        if valid:
            print(
                f"Rank {rank}: Validation passed! Distributed GEMM matches full GEMM."
            )
        else:
            print(f"Rank {rank}: Validation failed! Results do not match.")


    dist.barrier()

    if benchmark:
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        ms = iris.do_bench(run_experiment, dist.barrier)
        flops = perf(ms)
        print(f"Rank {rank}: {ms:.3f} ms  {perf(ms):.3f} tflops")
        json_writer.add_field("ms", ms)
        json_writer.add_field("flops", flops)

    dist.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    dist.destroy_process_group()

    
    

if __name__ == "__main__":
    main()