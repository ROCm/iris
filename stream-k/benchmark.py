import torch
import triton
import random
import sys
import os
import argparse
import json
from utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import iris

from communication import all_scatter_kernel, one_shot_kernel, all_reduce_kernel
from matmul_wrapper import matmul
from validation import validate_gemm

torch.manual_seed(123)
random.seed(123)


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

    def flush(self):
        self._write_to_file()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", type=int, default=4864, help="Number of rows in matrix A")
    parser.add_argument(
        "-n", type=int, default=4096, help="Number of columns in matrix B"
    )
    parser.add_argument(
        "-k", type=int, default=8256, help="Common dimension between matrices A and B"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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
        "--algorithm",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "all_scatter", "one_shot"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=256, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=32, help="Block size K")
    parser.add_argument(
        "--COMMUNICATION_TILE_M",
        type=int,
        default=128,
        help="M tile size for reduction, scatter or one-shot",
    )
    parser.add_argument(
        "--COMMUNICATION_TILE_N",
        type=int,
        default=128,
        help="N tile size for reduction, scatter or one-shot",
    )
    parser.add_argument("--gsize_m", type=int, default=8, help="Grid size M")
    parser.add_argument("--two_tiles", type=str, default="True", help="Use two tiles")
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps")
    parser.add_argument(
        "--waves_per_eu", type=int, default=0, help="Waves per execution unit"
    )
    parser.add_argument(
        "--mfmaInstrSize", type=int, default=16, help="MFMA instruction size"
    )
    parser.add_argument("--kpack", type=int, default=2, help="K packing size")
    parser.add_argument(
        "--heap_size", type=int, default=1 << 32, help="Iris heap size"
    )
    parser.add_argument(
        "--streamk_sms", type=int, default=256, help="Number of SMs for Stream-K"
    )
    parser.add_argument(
        "--total_sms", type=int, default=304, help="Total number of SMs"
    )
    parser.add_argument(
        "--communication_block_size", type=int, default=256, help="Communication block size"
    )

    return vars(parser.parse_args())


def main():
    args = parse_args()

    shmem = iris.Iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # GEMM
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "int8":
        datatype = torch.int8
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    assert (
        args["n"] % world_size == 0
    ), f"N ({args['n']}) must be divisible by world size ({world_size})."
    assert (
        args["k"] % world_size == 0
    ), f"K ({args['k']}) must be divisible by world size ({world_size})."

    A = shmem.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = shmem.randn(args["n"], args["k"], device="cuda", dtype=datatype).T
    C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    json_writer = JSONWriter(args["output_file"])

    for key, value in args.items():
        json_writer.add_field(key, value)

    args["M"] = args["m"]
    args["N"] = args["n"]
    args["K"] = args["k"]

    json_writer.add_field("m", args["m"])
    json_writer.add_field("n", args["n"])
    json_writer.add_field("k", args["k"])
    json_writer.add_field("algorithm", args["algorithm"])
    json_writer.add_field("world_size", world_size)

    # Splitting
    if args["algorithm"] == "all_scatter":
        args["n"] = args["n"] // world_size
        local_B = B[:, rank * args["n"] : (rank + 1) * args["n"]].clone()
        local_A = A
    elif args["algorithm"] == "all_reduce" or args["algorithm"] == "one_shot":
        rows_per_gpu = args["k"] // world_size
        start_row = rank * rows_per_gpu
        end_row = start_row + rows_per_gpu
        local_B = B[start_row:end_row, :]
        local_A = A[:, start_row:end_row]
    else:
        print("Unknown algorithm.")
        exit(1)
    global_C = shmem.zeros((args["M"], args["N"]), device="cuda", dtype=A.dtype)
    local_C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    if args["streamk_sms"] >= args["total_sms"]:
        print(
            f"Invalid number of stream-K SMs. {args['streamk_sms']} >= {args['total_sms']}"
        )
        exit(1)

    communication_sms = args["total_sms"] - args["streamk_sms"]

    communication_num_threads = args["communication_block_size"] * communication_sms
    grid = lambda meta: (triton.cdiv(communication_num_threads, meta["BLOCK_SIZE"]),)

    locks = shmem.zeros((args["streamk_sms"],), device="cuda", dtype=torch.int32)
    tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    P = shmem.zeros(
        (args["streamk_sms"], args["BLK_M"] * args["BLK_N"]),
        device="cuda",
        dtype=torch.float32,
    )
    bias = None

    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    json_writer.add_field("communication_sms", communication_sms)
    json_writer.add_field("streamk_sms", args["streamk_sms"])

    comm_registers = 0
    comm_spills = 0

    kernel_timing = {
        "streamk": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
        "communication": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    COMMUNICATION_ALGORITHM = NONE
    if args["algorithm"] == "one_shot":
        COMMUNICATION_ALGORITHM = ONE_SHOT
    elif args["algorithm"] == "all_reduce":
        COMMUNICATION_ALGORITHM = ALL_REDUCE
    elif args["algorithm"] == "all_scatter":
        COMMUNICATION_ALGORITHM = ALL_SCATTER

    def run_experiment():
        nonlocal local_C
        nonlocal global_C
        nonlocal comm_registers
        nonlocal comm_spills
        nonlocal kernel_timing
        torch.cuda.nvtx.range_push(f"GEMM + Communication")
        torch.cuda.nvtx.range_push(f"GEMM")
        with torch.cuda.stream(gemm_stream):
            kernel_timing["streamk"]["start_event"].record()
            local_C = matmul.apply(
                local_A,
                local_B,
                local_C,
                bias,
                P,
                locks,
                tile_completed,
                rank,
                world_size,
                args["streamk_sms"],
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                args["two_tiles"],
                args["num_stages"],
                args["num_warps"],
                args["waves_per_eu"],
                args["mfmaInstrSize"],
                args["kpack"],
                shmem.get_heap_bases(),
                COMMUNICATION_ALGORITHM,
            )
            kernel_timing["streamk"]["end_event"].record()
            kernel_timing["streamk"]["experiments"] += 1

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"Communication")
        with torch.cuda.stream(comm_stream):
            kernel_timing["communication"]["start_event"].record()
            if args["algorithm"] == "all_scatter":
                ss = all_scatter_kernel[grid](
                    local_C,
                    global_C,
                    tile_completed,
                    args["m"],
                    args["n"],
                    local_C.stride(0),
                    local_C.stride(1),
                    C.stride(0),
                    C.stride(1),
                    args["BLK_M"],
                    args["BLK_N"],
                    args["gsize_m"],
                    total_tiles,
                    args["communication_block_size"],
                    communication_sms,
                    shmem.get_heap_bases(),
                    rank,
                    world_size,
                    args["COMMUNICATION_TILE_M"],
                    args["COMMUNICATION_TILE_N"],
                )
            elif args["algorithm"] == "all_reduce":
                ss = all_reduce_kernel[grid](
                    local_C,
                    global_C,
                    tile_completed,
                    args["m"],
                    args["n"],
                    local_C.stride(0),
                    local_C.stride(1),
                    C.stride(0),
                    C.stride(1),
                    args["BLK_M"],
                    args["BLK_N"],
                    args["gsize_m"],
                    total_tiles,
                    args["communication_block_size"],
                    communication_sms,
                    shmem.get_heap_bases(),
                    rank,
                    world_size,
                    args["COMMUNICATION_TILE_M"],
                    args["COMMUNICATION_TILE_N"],
                )
            elif args["algorithm"] == "one_shot":
                ss = one_shot_kernel[grid](
                    local_C,
                    global_C,
                    tile_completed,
                    args["m"],
                    args["n"],
                    local_C.stride(0),
                    local_C.stride(1),
                    C.stride(0),
                    C.stride(1),
                    args["BLK_M"],
                    args["BLK_N"],
                    args["gsize_m"],
                    total_tiles,
                    args["communication_block_size"],
                    communication_sms,
                    shmem.get_heap_bases(),
                    rank,
                    world_size,
                    args["COMMUNICATION_TILE_M"],
                    args["COMMUNICATION_TILE_N"],
                )
            kernel_timing["communication"]["end_event"].record()
            kernel_timing["communication"]["experiments"] += 1

            comm_registers = ss.n_regs
            comm_spills = ss.n_spills
            shmem.log_debug(
                f"Communication kernel: {ss.n_regs} registers used, {ss.n_spills} spills"
            )

        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        for k in ["streamk", "communication"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(
                kernel_timing[k]["end_event"]
            )
            kernel_timing[k]["ms"] += ms

        shmem.barrier()
        torch.cuda.nvtx.range_pop()

    run_experiment()

    for k in ["streamk", "communication"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    streamk_registers = matmul.streamk_registers
    streamk_spills = matmul.streamk_spills

    json_writer.add_field("comm_registers", comm_registers)
    json_writer.add_field("comm_spills", comm_spills)
    json_writer.add_field("streamk_registers", streamk_registers)
    json_writer.add_field("streamk_spills", streamk_spills)

    if args["validate"]:
        matmul.set_debug(False)
        success = validate_gemm(A, B, global_C, shmem)
        json_writer.add_field("success", success)

        shmem.log("Validation passed.")

    shmem.barrier()

    if args["benchmark"]:
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        triton_ms = triton.testing.do_bench(run_experiment)
        triton_tflops = perf(triton_ms)
        algo_string = args["algorithm"]
        shmem.log_stats(
            f"tile matmul + {algo_string} (grid={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops"
        )

        json_writer.add_field("triton_tflops", triton_tflops)
        json_writer.add_field("triton_ms", triton_ms)

        for k in ["streamk", "communication"]:
            json_writer.add_field(
                k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"]
            )
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

    shmem.barrier()

    if rank == 0:
        json_writer.flush()

    shmem.barrier()

if __name__ == "__main__":
    main()
