---
name: linex-profiling
description: Profile GPU kernels at source-line granularity with cycle-level timing and stall analysis. Use when identifying performance hotspots at the source code level or analyzing instruction-level metrics mapped to source lines.
---

# Linex: Source-Level GPU Performance Profiling

Map GPU performance metrics to your source code lines. Get cycle-level timing, stall analysis, and instruction-level metrics for each line of source code.

## When to Use

- User asks to profile a GPU application at source-line granularity
- Need to identify which specific lines of code are performance bottlenecks
- Analyzing stall patterns and execution bottlenecks at the source level
- Understanding cycle-level timing for each line of code
- Instruction-level analysis mapped to source lines

## Instructions

1. **Ensure the target runs on AMD ROCm 7.0+** with `rocprofv3` available.
2. **Kernels must be compiled with `-g`** (debug symbols) for source mapping.
3. **Choose execution path:**
   - If a Linex MCP server is available, use its MCP tools:
     - `profile_application` to run and profile a target application with the options below.
     - `analyze_instruction_hotspots` to perform instruction-level hotspot analysis on collected profiles.
   - Otherwise use the Python API from the environment where Linex is installed.

### Python API

```python
from linex import Linex

profiler = Linex(
    target_cu=0,                      # Target compute unit
    shader_engine_mask="0xFFFFFFFF",  # All shader engines
    activity=10,                      # Activity counter polling
)

profiler.profile("./my_app", kernel_filter="my_kernel")

# Show hotspots (sorted by total_cycles)
for line in profiler.source_lines[:5]:
    print(f"{line.file}:{line.line_number}")
    print(f"  {line.total_cycles:,} cycles ({line.stall_percent:.1f}% stalled)")
    print(f"  Executed {line.execution_count} times")

# Find memory-bound lines
memory_bound = [
    l for l in profiler.source_lines 
    if l.stall_percent > 50
]

# Instruction-level analysis
for line in profiler.source_lines[:1]:
    for inst in line.instructions:
        print(f"{inst.isa}: {inst.latency_cycles} cycles")
```

### SourceLine Properties

- `file` - Source file path
- `line_number` - Line number
- `total_cycles` - Sum of all instruction cycles
- `stall_cycles` - Cycles spent waiting
- `idle_cycles` - Cycles slot was idle
- `execution_count` - Total executions
- `instructions` - List of ISA instructions
- `stall_percent` - Convenience: stall_cycles / total_cycles * 100

### InstructionData Properties

- `isa` - ISA instruction text
- `latency_cycles` - Total cycles for this instruction
- `stall_cycles` - Cycles spent waiting
- `idle_cycles` - Cycles slot was idle
- `execution_count` - How many times it ran
- `instruction_address` - Virtual address in GPU memory
- `file` - Parsed from source_location
- `line` - Parsed from source_location
- `stall_percent` - Convenience: stall_cycles / latency_cycles * 100

## Workflow

1. Ensure the target binary is built with `-g` (debug symbols) for source mapping.
2. Create a `Linex()` profiler; optionally set `target_cu`, `shader_engine_mask`, or `activity`.
3. Call `profiler.profile(command, kernel_filter=...)` to run profiling.
4. Access `profiler.source_lines` (sorted by total_cycles) to find hotspots.
5. Use `line.stall_percent` to identify memory-bound or dependency-bound lines.
6. Drill down into `line.instructions` for instruction-level analysis.
7. Use relative paths for the target binary so the skill is portable.

## Notes

- Requires ROCm 7.0+ with `rocprofv3` support.
- Source mapping requires kernels compiled with `-g` (debug symbols).
- `source_lines` are automatically sorted by `total_cycles` (descending).
- Use `kernel_filter` to profile specific kernels by name (regex pattern).
- For Triton or other frameworks, ensure debug symbols are available in the compiled output.
