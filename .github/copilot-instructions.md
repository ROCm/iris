# Iris: Multi-GPU Programming Framework

## Description

Iris is a Triton-based framework for Remote Memory Access (RMA) operations on AMD GPUs. It provides SHMEM-like APIs within Triton for Multi-GPU programming with:

- Clean abstractions with full symmetric heap implementation
- Pythonic PyTorch-like host APIs for tensor operations  
- Triton-style device APIs for load, store, and atomic operations
- Minimal dependencies (Triton, PyTorch, HIP runtime)
- Comprehensive examples showing communication/computation overlap

**FOLLOW THESE INSTRUCTIONS EXACTLY. Reference these instructions first before using search or bash commands.**

## GPU Environment

**You have AMD GPU access.** Your runner has 8x MI325X GPUs (gfx942) with ROCm installed. Do NOT skip GPU operations or say you lack GPU access. You can and should:

- Run benchmarks and post results (charts, tables) in PR comments
- Run tests with `torchrun --nproc_per_node=N` for multi-GPU tests
- Use `rocm-smi` to verify GPU status
- Run `python -c "import torch; print(torch.cuda.device_count())"` to confirm GPU count

When asked to run a benchmark, **run it and post the output**. Do not say you cannot.

### Running multi-GPU tests and benchmarks

Multi-GPU tests require `torch.distributed` initialization before pytest:
```bash
# Single GPU
pytest tests/unittests/ -v --tb=short

# Multi-GPU (N = number of GPUs)
torchrun --nproc_per_node=N -m pytest tests/ -v --tb=short

# Benchmarks use iris.bench framework
torchrun --nproc_per_node=8 benchmark/ops/bench_<name>.py
```

### iris.bench framework

Benchmarks use the declarative `iris.bench` framework. See existing `benchmark/ops/bench_*.py` files for examples. Output includes latency, throughput, and bandwidth tables. When posting benchmark results in PR comments, format as markdown tables.

## Prerequisites

- **GPU**: AMD GPUs with ROCm compatibility (tested on MI300X, MI325X, MI350X & MI355X)
- **ROCm/HIP Toolkit**: Required for building C++/HIP components
- **Docker/Apptainer**: Recommended for containerized development

## Build

iris is already installed in your environment via `pip install -e .` in the setup steps. You do not need to build or install anything. If you need to reinstall after modifying `setup.py` or C extensions:
```bash
pip install -e ".[dev]"
```

## Run

### Testing
```bash
# Run unit tests
pytest tests/unittests/

# Run example tests  
pytest tests/examples/

# Run specific example
python examples/00_load/load_bench.py
```

### Code Quality
```bash
# Linting and formatting
ruff check .
ruff format .

# Pre-commit validation (required)
ruff check . --fix
ruff format .
```

## Contributing Guidelines

### Development Workflow
1. **Setup**: Install with dev dependencies: `pip install -e ".[dev]"`
2. **Branch**: Create feature branch: `git checkout -b $USER/feature-name`
3. **Develop**: Follow existing code style, add tests, update docs
4. **Test**: Run `ruff check .`, `ruff format .`, and `pytest`
5. **Commit**: Use descriptive commit messages
6. **PR**: Create pull request with change details

### Code Standards
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Run pre-commit validation: `ruff check . --fix && ruff format .`

### Repository Structure
```
iris/
├── iris/                       # Main Python package
├── csrc/                       # C++/HIP source code
├── examples/                   # Algorithm implementations
├── tests/                      # Test suite
├── docker/                     # Docker configuration
└── docs/                      # Documentation
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
