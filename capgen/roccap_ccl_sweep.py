#!/usr/bin/env python3
"""
Read knob lists from YAML, run torchrun + roccap_wrapper for each Cartesian combo,
then rename .cap / *_heap_bases.json to include knob values in the filename.

Requires: PyYAML  (pip install pyyaml)

Shape knobs (two modes):
  - Paired: set `tensor_mn` (list of {m, n}) and `block_mn` (list of {block_size_m, block_size_n}).
    Sweeps (tensor_mn × block_mn × other args).
  - Legacy: omit both; put m, n, block_size_m, block_size_n in `args` as scalars or lists
    (full Cartesian product — all combinations).

`torchrun.nproc_per_node` may be a scalar or a list; each value is combined with the rest of the sweep.

Optional ``args.all_gather_variant`` / ``args.all_to_all_variant`` are forwarded when present (all-gather /
all-to-all examples). Omit keys to use example defaults. Remove keys your ``example`` does not support.

Example:
  cd iris/examples/25_ccl_all_gather
  python3 /path/to/roccap_ccl_sweep.py --config /path/to/roccap_ccl_sweep.example.yaml

Optional file logging (stdout+stderr of each torchrun) via YAML ``logging.dir`` or ``--log-dir DIR``
(see ``logging`` in the example YAML). Use ``tee: true`` or ``--log-tee`` to mirror output to the terminal.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

try:
    import yaml
except ImportError:
    print("Install PyYAML:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# Keys swept only via args (not tensor_mn / block_mn); all required in paired mode when present in YAML.
OTHER_ARG_KEYS: list[tuple[str, str]] = [
    ("comm_sms", "--comm_sms"),
    ("cache_modifier", "--cache_modifier"),
    ("num_warps", "--num_warps"),
    ("datatype", "--datatype"),
]

# Optional example.py flags: omit key in YAML → not passed (example defaults apply).
OPTIONAL_SCRIPT_ARGS: list[tuple[str, str]] = [
    ("all_gather_variant", "--all_gather_variant"),
    ("all_to_all_variant", "--all_to_all_variant"),
]

# Legacy: all eight passed through example.py
LEGACY_ARG_FLAGS: list[tuple[str, str]] = [
    ("m", "-m"),
    ("n", "-n"),
    ("block_size_m", "--block_size_m"),
    ("block_size_n", "--block_size_n"),
    ("comm_sms", "--comm_sms"),
    ("cache_modifier", "--cache_modifier"),
    ("num_warps", "--num_warps"),
    ("datatype", "--datatype"),
]

EXTRA_FLAGS: list[tuple[str, str]] = [
    ("num_stages", "--num_stages"),
    ("waves_per_eu", "--waves_per_eu"),
    ("heap_size", "--heap_size"),
]

# If torchrun fails, we still continue when every rank's .cap exists and is at least this size.
MIN_CAP_BYTES = 10 * 1024 * 1024


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _sanitize_cache(s: str) -> str:
    return re.sub(r'[.\'"\s]+', "", str(s)).replace(" ", "_") or "nomod"


def build_tag(kernel: str, combo: dict[str, Any], nproc: int) -> str:
    m, n = combo["m"], combo["n"]
    bm, bn = combo["block_size_m"], combo["block_size_n"]
    csms = combo["comm_sms"]
    slug = _sanitize_cache(str(combo["cache_modifier"]))
    dt = combo["datatype"]
    nw = combo["num_warps"]
    # Optional disambiguation (e.g. all_gather partitioned vs persistent)
    tag = f"{kernel}"
    # Optional collectives-specific variant (only one is usually set per example YAML)
    for _v in (combo.get("all_gather_variant"), combo.get("all_to_all_variant")):
        if _v is not None:
            tag += f"_{_v}"
    # Order: …64sms_1stage_cs_fp32_32warps_8nproc… (num_stages between sms and cache slug)
    tag += f"_{m}x{n}_{bm}x{bn}_{csms}sms"
    if combo.get("num_stages") is not None:
        tag += f"_{combo['num_stages']}stage"
    tag += f"_{slug}_{dt}_{nw}warps_{nproc}nproc"
    if combo.get("waves_per_eu") is not None:
        tag += f"_{combo['waves_per_eu']}wpe"
    if combo.get("heap_size") is not None:
        tag += f"_heap{combo['heap_size']}"
    return tag


def cap_files_meet_min_size(cwd: Path, kernel: str, nproc: int, min_bytes: int = MIN_CAP_BYTES) -> bool:
    """True if every {kernel}_rank_{r}.cap exists and is >= min_bytes."""
    for r in range(nproc):
        cap = cwd / f"{kernel}_rank_{r}.cap"
        if not cap.is_file():
            return False
        if cap.stat().st_size < min_bytes:
            return False
    return True


def run_torchrun_logged(
    cmd: list[str],
    cwd: Path,
    log_file: Path | None,
    tee: bool,
) -> subprocess.CompletedProcess[str]:
    """
    Run cmd; if log_file is set, append stdout+stderr to that file.
    If tee, also copy streams to the terminal (line-buffered where possible).
    """
    header = (
        f"# roccap_ccl_sweep {datetime.datetime.now().isoformat()}\n"
        f"# cwd: {cwd}\n"
        f"# {shlex.join(cmd)}\n\n"
    )

    if log_file is None:
        return subprocess.run(cmd, cwd=cwd)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8", errors="replace", buffering=1) as logf:
        logf.write(header)
        logf.flush()

        if tee:

            def pump_stream(stream: TextIO, *outs: TextIO) -> None:
                for line in stream:
                    for o in outs:
                        o.write(line)
                        o.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            reader = threading.Thread(
                target=pump_stream,
                args=(proc.stdout, logf, sys.stdout),
            )
            reader.start()
            proc.wait()
            reader.join()
            rc = proc.returncode if proc.returncode is not None else -1
            return subprocess.CompletedProcess(cmd, rc)
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


def describe_cap_problem(cwd: Path, kernel: str, nproc: int, min_bytes: int = MIN_CAP_BYTES) -> str:
    """Human-readable reason caps fail the min-size check (for error messages)."""
    for r in range(nproc):
        cap = cwd / f"{kernel}_rank_{r}.cap"
        if not cap.is_file():
            return f"missing {cap.name}"
        sz = cap.stat().st_size
        if sz < min_bytes:
            return f"{cap.name} is {sz} bytes (< {min_bytes} bytes)"
    return "unknown"


def rename_outputs(cwd: Path, kernel: str, tag: str, nproc: int, dry_run: bool) -> None:
    for r in range(nproc):
        cap = cwd / f"{kernel}_rank_{r}.cap"
        new_cap = cwd / f"{tag}_rank{r}.cap"
        if dry_run:
            print(f"mv -n {cap} {new_cap}")
        elif cap.is_file():
            cap.rename(new_cap)

        pattern = f"*_rank_{r}_heap_bases.json"
        for hb in cwd.glob(pattern):
            new_hb = cwd / f"{tag}_rank{r}_heap_bases.json"
            if dry_run:
                print(f"mv -n {hb} {new_hb}")
            else:
                hb.rename(new_hb)


def _parse_paired_shapes(cfg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return list of {m, n, block_size_m, block_size_n} base dicts, or None for legacy mode."""
    tm = cfg.get("tensor_mn")
    bm = cfg.get("block_mn")
    if tm is None and bm is None:
        return None
    if not tm or not bm:
        print(
            "For paired shapes, set both tensor_mn and block_mn (non-empty lists). "
            "Omit both to use legacy args.m / args.n / args.block_size_*.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not isinstance(tm, list) or not isinstance(bm, list):
        print("tensor_mn and block_mn must be YAML lists.", file=sys.stderr)
        sys.exit(1)

    bases: list[dict[str, Any]] = []
    for i, t in enumerate(tm):
        if not isinstance(t, dict) or "m" not in t or "n" not in t:
            print(f"tensor_mn[{i}] must be a mapping with keys m and n", file=sys.stderr)
            sys.exit(1)
    for i, b in enumerate(bm):
        if not isinstance(b, dict) or "block_size_m" not in b or "block_size_n" not in b:
            print(
                f"block_mn[{i}] must be a mapping with keys block_size_m and block_size_n",
                file=sys.stderr,
            )
            sys.exit(1)

    for t in tm:
        for b in bm:
            bases.append(
                {
                    "m": t["m"],
                    "n": t["n"],
                    "block_size_m": b["block_size_m"],
                    "block_size_n": b["block_size_n"],
                }
            )
    return bases


def main() -> None:
    ap = argparse.ArgumentParser(description="YAML-driven roccap / torchrun sweep")
    ap.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).with_name("roccap_ccl_sweep.example.yaml"),
        help="YAML config path",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands and renames only")
    ap.add_argument(
        "--log-dir",
        default=None,
        metavar="DIR",
        help="Write each run's stdout+stderr to DIR/<tag>.log (overrides YAML logging.dir)",
    )
    ap.add_argument(
        "--log-tee",
        action="store_true",
        help="With file logging, also print child output to the terminal",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    workdir = cfg.get("workdir") or ""
    tr = cfg.get("torchrun") or {}
    nproc_raw = tr.get("nproc_per_node", 8)
    nproc_list = [int(x) for x in _as_list(nproc_raw)]
    if not nproc_list:
        print("torchrun.nproc_per_node must be a non-empty scalar or list of integers", file=sys.stderr)
        sys.exit(1)
    standalone = tr.get("standalone", True)

    paths = cfg.get("paths") or {}
    wrapper = paths.get("wrapper", "../../scripts/roccap_wrapper.py")
    example_py = paths.get("example", "example.py")

    kernels = _as_list(cfg.get("kernel", "persistent_all_gather"))

    raw_args = cfg.get("args") or {}
    raw_extra = cfg.get("extra") or {}
    rename_enabled = (cfg.get("rename") or {}).get("enabled", True)

    log_cfg = cfg.get("logging") or {}
    yaml_log_dir = (log_cfg.get("dir") or "").strip()
    yaml_log_tee = bool(log_cfg.get("tee", False))
    effective_log_dir = args.log_dir if args.log_dir is not None else yaml_log_dir
    effective_log_tee = args.log_tee or yaml_log_tee

    paired_bases = _parse_paired_shapes(cfg)

    # Extra dimensions (num_stages, etc.)
    extra_dims: list[tuple[str, list[Any]]] = []
    for key, _flag in EXTRA_FLAGS:
        lst = _as_list(raw_extra.get(key))
        if not lst:
            extra_dims.append((key, [None]))
        else:
            extra_dims.append((key, lst))

    if paired_bases is not None:
        forbidden = {"m", "n", "block_size_m", "block_size_n"}
        bad = forbidden.intersection(raw_args.keys())
        if bad:
            print(
                f"In paired mode (tensor_mn / block_mn), remove from args: {sorted(bad)}",
                file=sys.stderr,
            )
            sys.exit(1)
        other_dims: list[tuple[str, list[Any]]] = []
        for key, _flag in OTHER_ARG_KEYS:
            if key not in raw_args:
                print(f"YAML must define args.{key} (scalar or list)", file=sys.stderr)
                sys.exit(1)
            other_dims.append((key, _as_list(raw_args[key])))
        for key, _flag in OPTIONAL_SCRIPT_ARGS:
            if key not in raw_args:
                other_dims.append((key, [None]))
            else:
                other_dims.append((key, _as_list(raw_args[key])))
        extra_keys = [d[0] for d in extra_dims]
        extra_lists = [d[1] for d in extra_dims]
        other_keys = [d[0] for d in other_dims]
        other_lists = [d[1] for d in other_dims]

        def iter_combos():
            for base in paired_bases:
                for ov in itertools.product(*other_lists):
                    for ev in itertools.product(*extra_lists):
                        row = {**base, **dict(zip(other_keys, ov)), **dict(zip(extra_keys, ev))}
                        yield row

    else:
        # Legacy: full Cartesian on all eight args + extra
        legacy_dims: list[tuple[str, list[Any]]] = []
        for key, _flag in LEGACY_ARG_FLAGS:
            if key not in raw_args:
                print(f"YAML must define args.{key} (scalar or list), or use tensor_mn + block_mn", file=sys.stderr)
                sys.exit(1)
            legacy_dims.append((key, _as_list(raw_args[key])))
        legacy_dims.extend(extra_dims)
        for key, _flag in OPTIONAL_SCRIPT_ARGS:
            if key not in raw_args:
                legacy_dims.append((key, [None]))
            else:
                legacy_dims.append((key, _as_list(raw_args[key])))
        keys = [d[0] for d in legacy_dims]
        value_lists = [d[1] for d in legacy_dims]

        def iter_combos():
            for combo_vals in itertools.product(*value_lists):
                yield dict(zip(keys, combo_vals))

    start = Path.cwd().resolve()
    cwd = (start / workdir).resolve() if workdir else start

    wrapper_path = Path(wrapper)
    if not wrapper_path.is_absolute():
        wrapper_path = (cwd / wrapper).resolve()
    if not args.dry_run and not wrapper_path.is_file():
        print(f"Wrapper not found: {wrapper_path}", file=sys.stderr)
        sys.exit(1)

    arg_flags_order = (
        (LEGACY_ARG_FLAGS if paired_bases is None else [
            ("m", "-m"),
            ("n", "-n"),
            ("block_size_m", "--block_size_m"),
            ("block_size_n", "--block_size_n"),
        ] + OTHER_ARG_KEYS)
        + OPTIONAL_SCRIPT_ARGS
    )

    for nproc in nproc_list:
        for kernel in kernels:
            for combo in iter_combos():
                opt: dict[str, Any] = {k: v for k, v in combo.items() if v is not None}

                tag = build_tag(str(kernel), opt, nproc)

                cmd: list[str] = ["torchrun", f"--nproc_per_node={nproc}"]
                if standalone:
                    cmd.append("--standalone")
                cmd.append(str(wrapper_path))
                cmd.extend(["-k", str(kernel), example_py])

                for key, flag in arg_flags_order:
                    if key not in opt:
                        continue
                    cmd.append(flag)
                    cmd.append(str(opt[key]))

                for key, flag in EXTRA_FLAGS:
                    if key not in opt:
                        continue
                    cmd.append(flag)
                    cmd.append(str(opt[key]))

                if args.dry_run:
                    print(shlex.join(cmd))
                    if effective_log_dir:
                        log_path = (cwd / effective_log_dir / f"{tag}.log").resolve()
                        print(f"  # log -> {log_path}")
                    if rename_enabled:
                        rename_outputs(cwd, str(kernel), tag, nproc, dry_run=True)
                    continue

                log_path: Path | None = None
                if effective_log_dir:
                    log_path = (cwd / effective_log_dir / f"{tag}.log").resolve()

                print("Running:", shlex.join(cmd), flush=True)
                if log_path is not None:
                    print(f"  log: {log_path}", flush=True)
                proc = run_torchrun_logged(cmd, cwd, log_path, tee=effective_log_tee)
                if proc.returncode != 0:
                    cmd_str = shlex.join(cmd)
                    if cap_files_meet_min_size(cwd, str(kernel), nproc):
                        print(
                            f"WARNING: Command failed (exit {proc.returncode}) but all rank .cap files "
                            f"are present and >= {MIN_CAP_BYTES // (1024 * 1024)} MiB; continuing sweep.\n"
                            f"  {cmd_str}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if rename_enabled:
                            rename_outputs(cwd, str(kernel), tag, nproc, dry_run=False)
                    else:
                        detail = describe_cap_problem(cwd, str(kernel), nproc)
                        print(
                            f"ERROR: Command failed (exit {proc.returncode}); {detail}.\n"
                            f"  {cmd_str}",
                            file=sys.stderr,
                            flush=True,
                        )
                        sys.exit(1)
                elif rename_enabled:
                    rename_outputs(cwd, str(kernel), tag, nproc, dry_run=False)


if __name__ == "__main__":
    main()
