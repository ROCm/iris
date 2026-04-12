# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for iris.tracing.kernel_artifacts (no GPU required)."""

import hashlib
import json
import types

import pytest

from iris.tracing import kernel_artifacts


# ---------------------------------------------------------------------------
# Fake CompiledKernel for testing without Triton
# ---------------------------------------------------------------------------
def _make_compiled(
    asm=None,
    constexprs=None,
    num_warps=4,
    num_stages=1,
    shared=0,
    kernel_hash="abc123",
    arg_names=None,
):
    """Build a minimal fake CompiledKernel."""
    if asm is None:
        asm = {"amdgcn": "s_endpgm", "ttir": "module {}", "ttgir": "module {}", "llir": "define void @kernel()"}

    meta = types.SimpleNamespace(num_warps=num_warps, num_stages=num_stages, shared=shared)

    compiled = types.SimpleNamespace(asm=asm, metadata=meta, hash=kernel_hash)

    if constexprs is not None:
        fn = types.SimpleNamespace(arg_names=arg_names or [])
        compiled.src = types.SimpleNamespace(constants=constexprs, fn=fn)

    return compiled


# ---------------------------------------------------------------------------
# _codegen_hash
# ---------------------------------------------------------------------------
class TestCodegenHash:
    def test_hashes_amdgcn_text(self):
        compiled = _make_compiled(asm={"amdgcn": "s_endpgm"})
        h = kernel_artifacts._codegen_hash(compiled)
        expected = hashlib.sha256(b"s_endpgm").hexdigest()[:12]
        assert h == expected

    def test_hashes_amdgcn_bytes(self):
        compiled = _make_compiled(asm={"amdgcn": b"\x00\x01\x02"})
        h = kernel_artifacts._codegen_hash(compiled)
        expected = hashlib.sha256(b"\x00\x01\x02").hexdigest()[:12]
        assert h == expected

    def test_falls_back_to_hsaco(self):
        compiled = _make_compiled(asm={"hsaco": b"binary_blob"})
        h = kernel_artifacts._codegen_hash(compiled)
        expected = hashlib.sha256(b"binary_blob").hexdigest()[:12]
        assert h == expected

    def test_falls_back_to_compiled_hash(self):
        compiled = _make_compiled(asm={}, kernel_hash="deadbeef")
        h = kernel_artifacts._codegen_hash(compiled)
        expected = hashlib.sha256(b"deadbeef").hexdigest()[:12]
        assert h == expected

    def test_empty_asm_and_no_hash(self):
        compiled = _make_compiled(asm={}, kernel_hash=None)
        h = kernel_artifacts._codegen_hash(compiled)
        expected = hashlib.sha256(b"unknown").hexdigest()[:12]
        assert h == expected

    def test_different_asm_different_hash(self):
        h1 = kernel_artifacts._codegen_hash(_make_compiled(asm={"amdgcn": "v_add_f32 v0, v1, v2"}))
        h2 = kernel_artifacts._codegen_hash(_make_compiled(asm={"amdgcn": "v_mul_f32 v0, v1, v2"}))
        assert h1 != h2


# ---------------------------------------------------------------------------
# _build_spec_dirname
# ---------------------------------------------------------------------------
class TestBuildSpecDirname:
    def test_block_sizes_and_warps(self):
        constexprs = {(0,): 128, (1,): 256}
        arg_names = ["BLOCK_SIZE_M", "BLOCK_SIZE_N"]
        compiled = _make_compiled(constexprs=constexprs, arg_names=arg_names, num_warps=8)
        name = kernel_artifacts._build_spec_dirname(compiled, dtype=None)
        assert name == "BM128_BN256_w8"

    def test_with_dtype(self):
        constexprs = {(0,): 64}
        arg_names = ["BLOCK_SIZE_M"]
        compiled = _make_compiled(constexprs=constexprs, arg_names=arg_names, num_warps=4)

        class FakeDtype:
            def __str__(self):
                return "torch.float16"

        name = kernel_artifacts._build_spec_dirname(compiled, dtype=FakeDtype())
        assert "fp16" in name
        assert "BM64" in name

    def test_fallback_to_hash(self):
        compiled = _make_compiled(constexprs={}, arg_names=[], kernel_hash="0123456789abcdef")
        # Remove num_warps so no parts are generated
        compiled.metadata = types.SimpleNamespace(num_stages=1, shared=0)
        name = kernel_artifacts._build_spec_dirname(compiled, dtype=None)
        assert name == "0123456789ab"

    def test_with_block_k(self):
        constexprs = {(0,): 128, (1,): 256, (2,): 32}
        arg_names = ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]
        compiled = _make_compiled(constexprs=constexprs, arg_names=arg_names, num_warps=4)
        name = kernel_artifacts._build_spec_dirname(compiled, dtype=None)
        assert name == "BM128_BN256_BK32_w4"


# ---------------------------------------------------------------------------
# _dtype_short_name
# ---------------------------------------------------------------------------
class TestDtypeShortName:
    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("torch.float16", "fp16"),
            ("torch.bfloat16", "bf16"),
            ("torch.float32", "fp32"),
            ("torch.int8", "i8"),
            ("torch.float8_e4m3fnuz", "fp8e4m3"),
        ],
    )
    def test_known_dtypes(self, dtype_str, expected):
        class FakeDtype:
            def __str__(self):
                return dtype_str

        assert kernel_artifacts._dtype_short_name(FakeDtype()) == expected

    def test_none(self):
        assert kernel_artifacts._dtype_short_name(None) is None

    def test_unknown_strips_torch_prefix(self):
        class FakeDtype:
            def __str__(self):
                return "torch.complex64"

        assert kernel_artifacts._dtype_short_name(FakeDtype()) == "complex64"


# ---------------------------------------------------------------------------
# Dedup / _save end-to-end
# ---------------------------------------------------------------------------
class TestDedup:
    def test_second_save_is_skipped(self, tmp_path):
        """Calling _save twice with the same compiled kernel should only write once."""
        original_dir = kernel_artifacts._artifacts_dir
        original_enabled = kernel_artifacts._enabled
        try:
            kernel_artifacts._artifacts_dir = tmp_path
            kernel_artifacts._enabled = True

            compiled = _make_compiled()
            kernel_artifacts._save(compiled, "test_algo", "test_kernel", 0, None, (64,))

            # Find the metadata file
            metadata_files = list(tmp_path.rglob("metadata.json"))
            assert len(metadata_files) == 1

            # Record mtime
            mtime = metadata_files[0].stat().st_mtime

            # Save again — should be deduped
            kernel_artifacts._save(compiled, "test_algo", "test_kernel", 0, None, (64,))

            metadata_files_after = list(tmp_path.rglob("metadata.json"))
            assert len(metadata_files_after) == 1
            assert metadata_files_after[0].stat().st_mtime == mtime
        finally:
            kernel_artifacts._artifacts_dir = original_dir
            kernel_artifacts._enabled = original_enabled

    def test_different_codegen_gets_separate_dir(self, tmp_path):
        """Different assembly should produce different codegen hash dirs."""
        original_dir = kernel_artifacts._artifacts_dir
        original_enabled = kernel_artifacts._enabled
        try:
            kernel_artifacts._artifacts_dir = tmp_path
            kernel_artifacts._enabled = True

            compiled1 = _make_compiled(asm={"amdgcn": "v_add_f32 v0, v1, v2"})
            compiled2 = _make_compiled(asm={"amdgcn": "v_mul_f32 v0, v1, v2"})

            kernel_artifacts._save(compiled1, "test_algo", "test_kernel", 0, None, (64,))
            kernel_artifacts._save(compiled2, "test_algo", "test_kernel", 0, None, (64,))

            metadata_files = list(tmp_path.rglob("metadata.json"))
            assert len(metadata_files) == 2
        finally:
            kernel_artifacts._artifacts_dir = original_dir
            kernel_artifacts._enabled = original_enabled


# ---------------------------------------------------------------------------
# _write_artifacts
# ---------------------------------------------------------------------------
class TestWriteArtifacts:
    def test_writes_all_ir_files(self, tmp_path):
        asm = {
            "ttir": "module { func @kernel() }",
            "ttgir": "module { tt.func @kernel() }",
            "llir": "define void @kernel() { ret void }",
            "amdgcn": "s_endpgm",
        }
        compiled = _make_compiled(asm=asm)
        metadata = {"kernel_name": "test", "codegen_hash": "abc"}

        kernel_artifacts._write_artifacts(tmp_path, compiled, metadata)

        assert (tmp_path / "kernel.ttir").exists()
        assert (tmp_path / "kernel.ttgir").exists()
        assert (tmp_path / "kernel.llir").exists()
        assert (tmp_path / "kernel.amdgcn").exists()
        assert (tmp_path / "metadata.json").exists()

        meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        assert meta["kernel_name"] == "test"

    def test_missing_asm_keys_skipped(self, tmp_path):
        """Only available IR levels should be written."""
        compiled = _make_compiled(asm={"amdgcn": "s_endpgm"})
        metadata = {"kernel_name": "test"}

        kernel_artifacts._write_artifacts(tmp_path, compiled, metadata)

        assert (tmp_path / "kernel.amdgcn").exists()
        assert not (tmp_path / "kernel.ttir").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_binary_asm_written_as_bytes(self, tmp_path):
        compiled = _make_compiled(asm={"amdgcn": b"\x00\x01\x02\x03"})
        metadata = {"kernel_name": "test"}

        kernel_artifacts._write_artifacts(tmp_path, compiled, metadata)

        assert (tmp_path / "kernel.amdgcn").read_bytes() == b"\x00\x01\x02\x03"
