# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
import torch

import json
import numpy as np
import os

# Communication Algorithms
NONE = tl.constexpr(0)  # TODO: None is bad here
ALL_SCATTER = tl.constexpr(1)
ALL_REDUCE = tl.constexpr(2)
ONE_SHOT = tl.constexpr(3)
ONE_SHOT_V1 = tl.constexpr(4)
ONE_SHOT_V2 = tl.constexpr(5)
ALL_GATHER = tl.constexpr(6)


dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    try:
        return dtype_map[datatype]
    except KeyError:
        print(f"Unknown datatype: {datatype}")
        exit(1)


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    return list(dtype_map.keys())[list(dtype_map.values()).index(dtype)]


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
        self.mm_begin_timestamp = torch.empty(num_tiles, dtype=torch.int64, device="cuda")
        self.mm_end_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")

        self.comm_begin_timestamp = torch.empty(num_tiles, dtype=torch.int64, device="cuda")
        self.comm_middle_min_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")
        self.comm_middle_max_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")
        self.comm_end_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")

    def reset(self):
        self.mm_begin_timestamp.fill_(self.max_ts)
        self.mm_end_timestamp.fill_(self.min_ts)

        self.comm_begin_timestamp.fill_(self.max_ts)
        self.comm_middle_min_timestamp.fill_(self.max_ts)
        self.comm_middle_max_timestamp.fill_(self.min_ts)
        self.comm_end_timestamp.fill_(self.min_ts)

    def to_json(self, filename, gpu_freq):
        cycles_to_us = lambda cycles: cycles / gpu_freq

        gemm_begin_us = cycles_to_us(self.mm_begin_timestamp.cpu().numpy())
        gemm_end_us = cycles_to_us(self.mm_end_timestamp.cpu().numpy())

        comm_begin_us = cycles_to_us(self.comm_begin_timestamp.cpu().numpy())
        poll_end_us = cycles_to_us(self.comm_middle_max_timestamp.cpu().numpy())
        op_begin_us = cycles_to_us(self.comm_middle_min_timestamp.cpu().numpy())
        op_end_us = cycles_to_us(self.comm_end_timestamp.cpu().numpy())

        min_timestamp = min(
            np.min(gemm_begin_us),
            np.min(gemm_end_us),
            np.min(comm_begin_us),
            np.min(poll_end_us),
            np.min(op_begin_us),
            np.min(op_end_us),
        )

        gemm_begin_us = gemm_begin_us - min_timestamp
        gemm_end_us = gemm_end_us - min_timestamp
        comm_begin_us = comm_begin_us - min_timestamp
        poll_end_us = poll_end_us - min_timestamp
        op_begin_us = op_begin_us - min_timestamp
        op_end_us = op_end_us - min_timestamp

        data = [
            {
                "tile_id": i,
                "gemm_begin": int(gemm_begin),
                "gemm_end": int(gemm_end),
                "poll_begin": int(comm_begin),
                "poll_end": int(poll_end),
                "op_begin": int(op_begin),
                "op_end": int(
                    op_end,
                ),
                "comm_begin": int(comm_begin),
                "comm_end": int(
                    op_end,
                ),
            }
            for i, (
                gemm_begin,
                gemm_end,
                comm_begin,
                poll_end,
                op_begin,
                op_end,
            ) in enumerate(
                zip(
                    gemm_begin_us,
                    gemm_end_us,
                    comm_begin_us,
                    poll_end_us,
                    op_begin_us,
                    op_end_us,
                )
            )
        ]
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


def is_triton_interpret_set():
    return "TRITON_INTERPRET" in os.environ


# Re-export device utility functions from iris module
# These are kept here for backward compatibility with existing examples
