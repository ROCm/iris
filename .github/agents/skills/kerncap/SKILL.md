---
name: test-kerncap
description: Test local kerncap changes end-to-end by profiling an application, extracting a kernel, and validating the reproducer. Use when the user asks to test kerncap against any HIP or Triton workload, or wants to validate extraction on a real GPU application.
---

# Test kerncap Against an Application

Test local kerncap changes end-to-end by extracting and validating a kernel from any application.

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `app_cmd` | **Yes** | Full command to run the application (binary + arguments), e.g. `$WORK/dev/llama.cpp/build/bin/llama-bench -m model.gguf -p 512 -n 32` |
| `conda_env` | No | Conda environment to activate before running commands (e.g. `llama_cpp`). If not provided, use the current environment. |
| `kernel_name` | No | Name of the kernel to extract (e.g. `mul_mat_q`). If not provided, profile the application first and select the top kernel by execution time. |

## Paths

| Item | Path |
|------|------|
| kerncap source | `kerncap/` (relative to IntelliKit repo root) |
| Output directory | `/tmp/kerncap-test/<kernel_name>` |

## Environment Setup

If `conda_env` is provided, activate it before any other step:

```bash
conda activate <conda_env>
```

If already in a different environment, switch explicitly. Do not assume the current shell environment is correct.

If `conda_env` is not provided, proceed with the current environment as-is.

## Workflow

### Step 1: Reinstall kerncap

Ensure the correct environment is active (if applicable), then uninstall and reinstall to pick up local changes:

```bash
pip uninstall kerncap -y && pip install kerncap/
```

### Step 2: Profile to identify target kernel

**If `kernel_name` was provided**: Skip this step and proceed to Step 3.

**If `kernel_name` was not provided**: Run profiling to discover the top bottleneck kernel:

```bash
kerncap profile -- <app_cmd>
```

Select the kernel with the highest total execution time from the profile output. Use its name as `kernel_name` for all subsequent steps. Tell the user which kernel was selected and why.

**Important**: Use a sufficiently long substring from the profile output as `kernel_name` so that `kerncap extract` matches the intended kernel, not a different instantiation. For example, templated kernels like `mul_mat_q` have many instantiations differing only by template parameters; passing just `mul_mat_q` will capture the first dispatch that matches, which may not be the top-ranked one. Prefer including template parameters in the substring (e.g. `mul_mat_q<(ggml_type)39` instead of `mul_mat_q`).

### Step 3: Extract the kernel

```bash
kerncap extract --help
```

Use the help output to construct the appropriate `kerncap extract` command for the application. Key flags to determine:

- `--cmd` — the application command (`app_cmd`)
- `--source-dir` — where the kernel source lives (ask the user if unclear)
- `--output` — `/tmp/kerncap-test/<kernel_name>`
- `--language` — `hip` or `triton` depending on the workload
- Any additional flags (`-D` defines, `--dispatch`, etc.)

**If extraction fails or produces errors**: Stop here and report the full error output. This indicates the local kerncap changes have a bug that needs fixing.

**If extraction succeeds**: Inspect the output directory for expected files (metadata.json, argument dumps, source files). If the output looks reasonable, proceed to compile and run.

### Step 4: Compile and run the reproducer

Navigate to the output directory and build/run the reproducer:

```bash
cd /tmp/kerncap-test/<kernel_name>
make run
```

**If `make run` fails**: Stop here and report the full compiler or runtime error output. This is the primary signal that kerncap generated an incorrect reproducer.

**If `make run` succeeds**: Proceed to validation.

### Step 5: Validate the reproducer

**5a. Smoke test** — confirm baseline replay works:

```bash
kerncap validate /tmp/kerncap-test/<kernel_name>
```

This is a smoke test only (VA-faithful captures). It confirms the replay runs without crashing but does not check numerical correctness.

**5b. Recompile** — build a baseline HSACO from the unmodified kernel source:

```bash
cd /tmp/kerncap-test/<kernel_name>
make recompile
```

This confirms the VFS-overlay recompile pipeline works. It produces `optimized.hsaco` from the unmodified `kernel_variant.cpp`.

**If `make recompile` fails**: Stop here and report the error. This indicates an issue with the source finder or VFS overlay generation.

**5c. Correctness validation** — compare recompiled HSACO against captured baseline:

```bash
kerncap validate /tmp/kerncap-test/<kernel_name> --hsaco /tmp/kerncap-test/<kernel_name>/optimized.hsaco
```

This runs replay twice (captured HSACO vs recompiled HSACO) and compares outputs byte-for-byte. Since the kernel source is unmodified, they should match exactly. A failure here indicates a recompilation fidelity issue.

### Step 6: Report results

Summarize:
- Whether reinstall succeeded
- Whether profiling identified a kernel (if applicable, and which one)
- Whether extraction completed (and any warnings)
- Whether `make run` compiled and executed successfully
- Whether smoke test passed (Step 5a)
- Whether recompile succeeded (Step 5b)
- Whether correctness validation passed (Step 5c)
- Any errors or warnings encountered at each step
