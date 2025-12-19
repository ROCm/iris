# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch

try:
    if torch.cuda.is_available():
        cu_count = torch.cuda.get_device_properties(0).multi_processor_count
    else:
        cu_count = 304  # Default for MI300
except Exception:
    cu_count = 304  # Default for MI300

# Sample input (replace with file read if needed)
config = {
    "world_size": 8,
    "m": 8192,
    "n": 576,
    "k": 36864,
    "datatype": "fp16",
    "algorithm": "all_scatter",
    "output_file": "log.json",
    "BLK_M": 256,
    "BLK_N": 64,
    "BLK_K": 64,
    "COMMUNICATION_TILE_M": 128,
    "COMMUNICATION_TILE_N": 128,
    "gsize_m": 6,
    "two_tiles": "True",
    "num_stages": 1,
    "num_warps": 8,
    "waves_per_eu": 0,
    "mfmaInstrSize": 16,
    "kpack": 2,
    "heap_size": 8589934592,
    "gemm_sms": 48,
    "total_sms": cu_count,
    "communication_block_size": 256,
    "communication_sms_multiplier": 1,
    "M": 8192,
    "N": 4608,
    "K": 36864,
    "communication_sms": 256,
    "comm_registers": 256,
    "comm_spills": 0,
    "gemm_registers": 98,
    "gemm_spills": 0,
    "triton_tflops": 326.5083188319966,
    "triton_ms": 8.52394455909729,
    "gemm_ms": 8.405209632146926,
    "gemm_experiments": 126,
    "communication_ms": 1.5411831916324676,
    "communication_experiments": 126,
}

# Calculate effective link bandwidth from communication time
# Each rank sends its full output to (world_size - 1) other ranks
bytes_per_elem = 2 if config["datatype"] == "fp16" else 4
data_bytes_per_rank = config["M"] * config["n"] * bytes_per_elem
total_data_sent_per_rank = data_bytes_per_rank
bandwidth_Gbps_per_rank = total_data_sent_per_rank / (config["communication_ms"] / 1000) / 1e9
bandwidth_Gbps_per_world = bandwidth_Gbps_per_rank * config["world_size"]

# Print summary
print(f"Algorithm                   : {config['algorithm']}")
print(f"World Size                  : {config['world_size']}")
print(f"Matrix Shape                : ({config['m']}, {config['n']}, {config['k']})")
print(f"TFLOPs                      : {config['triton_tflops']:.2f}")
print(f"Total Time (ms)             : {config['triton_ms']:.3f}")
print(f"GEMM Time (ms)              : {config['gemm_ms']:.3f}")
print(f"Comm Time (ms)              : {config['communication_ms']:.3f}")
print(f"Data Bytes per Rank         : {data_bytes_per_rank} bytes")
print(f"Average BW (GB/s)           : {bandwidth_Gbps_per_rank:.2f}")
print(f"Bytes (GB) per Rank         : {total_data_sent_per_rank / 1e9:.2f} GB")
print(f"BW (GB/s) Total             : {bandwidth_Gbps_per_world:.2f}")
