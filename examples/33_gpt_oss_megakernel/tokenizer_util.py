# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Minimal GPT-OSS tokenizer wrapper using the `tokenizers` library directly
(no `transformers` modeling dependency), loading the HF snapshot's tokenizer.json."""

from __future__ import annotations

import glob
import os


def _snapshot(snapshot: str | None) -> str:
    if snapshot and os.path.isdir(snapshot):
        return snapshot
    home = os.path.expanduser("~")
    base = os.path.join(home, ".cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots")
    cand = sorted(glob.glob(os.path.join(base, "*")))
    if not cand:
        raise FileNotFoundError(f"No gpt-oss-120b snapshot under {base}")
    return cand[-1]


class GptOssTokenizer:
    def __init__(self, snapshot: str | None = None):
        from tokenizers import Tokenizer

        path = os.path.join(_snapshot(snapshot), "tokenizer.json")
        self.tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


def load_tokenizer(snapshot: str | None = None) -> GptOssTokenizer:
    return GptOssTokenizer(snapshot)
