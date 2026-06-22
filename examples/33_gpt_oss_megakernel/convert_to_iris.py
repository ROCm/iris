# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Convert an OpenAI GPT-OSS-120B HuggingFace checkpoint into a single `.iris` weight
file consumable by the megakernel host driver, and keep it as a persistent artifact.

The `.iris` format is a flat, mmap-friendly tensor archive (little-endian):
  [0:8)    magic  b"IRISMDL1"
  [8:12)   u32 version (=1)
  [12:16)  u32 num_tensors
  [16:...) tensor table: for each tensor
             128-byte UTF-8 name (NUL padded)
             u64 offset (absolute file byte offset of blob)
             u64 nbytes
             u32 dtype  (0=bf16, 1=fp32, 2=uint8/mxfp4-raw, 3=int32)
             u32 ndim
             8 x u64 shape (unused dims = 0)
  header padded to 4096; each blob 256-byte aligned.

Experts are stored as RAW HF MXFP4 (blocks uint8 + scales uint8), NOT dequantized,
so the file stays ~63GB (fits) and the megakernel dequantizes on the fly. This is
the memory-correct choice (full BF16 expansion would be ~240GB).

Usage:
  python convert_to_iris.py --out /work/.../gptoss_120b.iris [--layers N]
  # then:
  python run_iris_megakernel.py --model /work/.../gptoss_120b.iris
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import torch

from reference import GptOssConfig
from load_hf import load_hf_weights

MAGIC = b"IRISMDL1"
VERSION = 1
HEADER_ALIGN = 4096
BLOB_ALIGN = 256
NAME_BYTES = 128
DT_BF16, DT_FP32, DT_U8, DT_I32 = 0, 1, 2, 3

_TORCH_DT = {DT_BF16: torch.bfloat16, DT_FP32: torch.float32, DT_U8: torch.uint8, DT_I32: torch.int32}
_NP_DT = {DT_BF16: np.uint16, DT_FP32: np.float32, DT_U8: np.uint8, DT_I32: np.int32}


def _dt_of(t: torch.Tensor) -> int:
    if t.dtype == torch.bfloat16:
        return DT_BF16
    if t.dtype == torch.float32:
        return DT_FP32
    if t.dtype == torch.uint8:
        return DT_U8
    if t.dtype in (torch.int32, torch.int64):
        return DT_I32
    raise ValueError(t.dtype)


def _to_np(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).cpu().numpy()
    if t.dtype == torch.int64:
        return t.to(torch.int32).cpu().numpy()
    return t.cpu().numpy()


def _align(n, a):
    return (n + a - 1) // a * a


def collect_tensors(cfg: GptOssConfig, L: int, snapshot: str | None):
    """Yield (name, torch.Tensor) for the whole model in megakernel-ready layout."""
    w = load_hf_weights(GptOssConfig(), snapshot=snapshot, num_layers=L, device="cpu", dtype=torch.bfloat16)
    out = {}
    out["embed"] = w.embed
    out["final_norm"] = w.final_norm
    out["lm_head"] = w.lm_head
    keys_f32 = ["norm_attn", "norm_moe", "sinks"]
    keys_bf16 = ["w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o", "router_w", "router_b", "gate_up_b", "down_b"]
    keys_u8 = ["gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales"]
    for l in range(L):
        lw = w.layers[l]
        for k in keys_f32:
            out[f"L{l}.{k}"] = lw[k].float()
        for k in keys_bf16:
            out[f"L{l}.{k}"] = lw[k].to(torch.bfloat16)
        for k in keys_u8:
            out[f"L{l}.{k}"] = lw[k].to(torch.uint8).contiguous()
    return out


def write_iris(path: str, tensors: dict[str, torch.Tensor]):
    names = list(tensors.keys())
    n = len(names)
    entry_sz = NAME_BYTES + 8 + 8 + 4 + 4 + 8 * 8
    header_sz = _align(16 + n * entry_sz, HEADER_ALIGN)

    # compute offsets
    offsets = {}
    cur = header_sz
    for nm in names:
        cur = _align(cur, BLOB_ALIGN)
        t = tensors[nm]
        offsets[nm] = cur
        cur += _to_np(t).nbytes

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", n))
        for nm in names:
            t = tensors[nm]
            arr = _to_np(t)
            nb = arr.nbytes
            dt = _dt_of(t)
            shp = list(t.shape) + [0] * (8 - t.dim())
            name_b = nm.encode("utf-8")[:NAME_BYTES].ljust(NAME_BYTES, b"\x00")
            f.write(name_b)
            f.write(struct.pack("<Q", offsets[nm]))
            f.write(struct.pack("<Q", nb))
            f.write(struct.pack("<I", dt))
            f.write(struct.pack("<I", t.dim()))
            f.write(struct.pack("<8Q", *shp))
        # pad header
        pad = header_sz - f.tell()
        f.write(b"\x00" * pad)
        # blobs
        for nm in names:
            blob_off = offsets[nm]
            cur = f.tell()
            if cur < blob_off:
                f.write(b"\x00" * (blob_off - cur))
            f.write(_to_np(tensors[nm]).tobytes())
    print(f"wrote {path}: {n} tensors, {os.path.getsize(path)/1e9:.1f} GB")


def read_iris_header(path: str):
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC, "bad magic"
        (ver,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<I", f.read(4))
        entries = {}
        for _ in range(n):
            name = f.read(NAME_BYTES).rstrip(b"\x00").decode("utf-8")
            off, nb = struct.unpack("<QQ", f.read(16))
            dt, nd = struct.unpack("<II", f.read(8))
            shape = struct.unpack("<8Q", f.read(64))[:nd]
            entries[name] = (off, nb, dt, tuple(shape))
    return ver, entries


def load_iris_tensor(path_or_mmap, entry, device="cuda"):
    off, nb, dt, shape = entry
    np_dt = _NP_DT[dt]
    arr = np.memmap(path_or_mmap, dtype=np_dt, mode="r", offset=off, shape=(nb // np.dtype(np_dt).itemsize,))
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    if dt == DT_BF16:
        t = t.view(torch.bfloat16)
    t = t.reshape(shape) if shape else t
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()
    cfg = GptOssConfig()
    L = args.layers if args.layers > 0 else cfg.num_layers
    print(f"collecting {L} layers from HF ...")
    tensors = collect_tensors(cfg, L, args.snapshot)
    write_iris(args.out, tensors)
    # sanity: reparse header
    ver, ents = read_iris_header(args.out)
    print(f"verified header v{ver}: {len(ents)} tensor entries")


if __name__ == "__main__":
    main()
