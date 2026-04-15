---
name: nexus-trace
description: Extract GPU kernel assembly and HIP source from HSA packet traces. Use when analyzing what code ran on the GPU, debugging kernel dispatch, or inspecting assembly and source mapping.
---

# Nexus: HSA Packet Source Code Extractor

Intercepts HSA packets from a running process and extracts, per kernel, assembly and HIP source into a structured trace (e.g. JSON). Use for kernel-level inspection and assembly/source correlation.

## When to Use

- User needs to see which kernels ran and their assembly or HIP source
- Debugging or analyzing GPU dispatch and code generation
- Inspecting assembly-to-source mapping for a HIP (or ROCm) application

## Instructions

1. **Ensure the target runs on AMD ROCm** and uses HSA (e.g. HIP application or ROCm runtime).
2. **Choose execution path:**
   - If a Nexus MCP server is available, use its tools: `list_kernels` to enumerate kernels in a trace, and `extract_kernel_code` to get assembly and HIP/source mapping (signature, files, lines). See `nexus/nexus/mcp/server.py` for tool parameters and schemas.
   - Otherwise use the Python API from the environment where Nexus is installed.

### Python API (recommended when no MCP)

```python
from nexus import Nexus

nexus = Nexus(log_level=1)
trace = nexus.run(["python", "my_gpu_script.py"])

# Or run a binary:
# trace = nexus.run(["./my_hip_app"])

for kernel in trace:
    print(kernel.name, len(kernel.assembly), "instructions")
    for i, asm_line in enumerate(kernel.assembly, 1):
        print(f"  {i}. {asm_line}")
    for line_no, hip_line in zip(kernel.lines or range(1, len(kernel.hip)+1), kernel.hip):
        print(f"  {line_no}: {hip_line}")

# Access by kernel name
k = trace["vector_add(float const*, float const*, float*, int)"]
print(k.assembly, k.hip, k.signature, k.files, k.lines)

# Save/load trace
trace.save("trace.json")
loaded = Nexus.load("trace.json")
```

Set `log_level` (0–4) to control verbosity. Use relative paths for the run command and output file so the skill is portable.

### Environment-based usage (no Python API)

When the process cannot be launched via `nexus.run()`:

1. Set `HSA_TOOLS_LIB` to the Nexus shared library path (e.g. `build/lib/libnexus.so` or the installed path).
2. Set `NEXUS_OUTPUT_FILE` to the output JSON path.
3. Set `NEXUS_LOG_LEVEL` (0–4) if needed.
4. Run the application as usual; it will be traced and the output file will contain the kernel data.

Optional: `NEXUS_EXTRA_SEARCH_PREFIX` (colon-separated) for HIP source search; `TRITON_DISABLE_LINE_INFO=0` for Triton kernel line info.

## Workflow

1. Identify the command that runs the GPU workload (e.g. `python script.py` or `./app`).
2. If using the Python API: create `Nexus(log_level=...)`, call `nexus.run([...])`, then iterate `trace` and optionally `trace.save(...)`.
3. If using the env method: set `HSA_TOOLS_LIB` and `NEXUS_OUTPUT_FILE`, then run the app; open the JSON and parse the `kernels` structure.
4. Use kernel `signature`, `assembly`, `hip`, `files`, and `lines` to analyze what ran and map assembly back to source.
5. Use relative paths for commands and output files.

## Notes

- Nexus is intended for research/analysis; ensure the target environment has the Nexus library and compatible ROCm/HSA stack.
- For Triton kernels, enable line info via `TRITON_DISABLE_LINE_INFO=0` when using the Python API.
