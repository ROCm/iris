# AGENTS.md

## Project Overview

Iris is a Triton-based framework for Remote Memory Access (RMA) operations on AMD GPUs. It provides SHMEM-like APIs within Triton for Multi-GPU programming with:

- Clean abstractions with a full symmetric heap implementation
- Pythonic PyTorch-like host APIs for tensor operations
- Triton-style device APIs for load, store, and atomic operations
- Minimal dependencies (Triton, PyTorch, HIP runtime)
- Comprehensive examples showing communication/computation overlap

**Supported GPUs**: MI300X, MI350X & MI355X (other ROCm-compatible AMD GPUs may work)

## Dev Environment Setup

### Prerequisites

- AMD GPU with ROCm 6.3.1+ support
- Python 3.10+, PyTorch 2.0+ (ROCm version)
- Triton (see [installation docs](docs/getting-started/installation.md) for the verified commit)
- Docker or Apptainer (recommended for containerized development)

### Docker Compose (Recommended)

```bash
# Build and start development container (takes 45-60 minutes — do NOT cancel)
docker compose up --build -d

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode
cd iris && pip install -e ".[dev]"
```

### Local Install

```bash
pip install -e ".[dev]"
```

> **Note**: Requires an AMD GPU with ROCm/HIP toolkit installed.

## Code Style

- Use `ruff` for linting and formatting (configured in `pyproject.toml`).
- Line length: 120 characters.
- Double quotes for strings.
- Run before every commit:

```bash
ruff check . --fix
ruff format .
```

## Testing Instructions

Tests require at least **2 AMD GPUs**. Use `torchrun` via the helper script:

```bash
# Run all unit tests (2 ranks)
python tests/run_tests_distributed.py tests/unittests/ --num_ranks 2 -v

# Run all example tests (2 ranks)
python tests/run_tests_distributed.py tests/examples/ --num_ranks 2 -v

# Run a single test file
python tests/run_tests_distributed.py tests/unittests/test_load_triton.py --num_ranks 2 -v
```

> **Environment note**: The test runner sets `HSA_NO_SCRATCH_RECLAIM=1` automatically, which is required for RCCL on ROCm.

### Without AMD GPUs

If you don't have AMD GPU access you can still contribute:

1. Edit code locally and push to a branch.
2. The CI pipeline (`.github/workflows/`) will run the full test suite against real hardware.
3. Run linting locally to catch style issues before CI:

```bash
ruff check . --fix && ruff format .
```

## Repository Structure

```
iris/
├── iris/                   # Main Python package
│   ├── ops/                # RMA operation kernels (load, store, atomics)
│   ├── ccl/                # Collective communication primitives
│   ├── experimental/       # Gluon-based experimental APIs
│   └── allocators/         # Symmetric heap allocators
├── csrc/                   # C++/HIP source code
├── examples/               # Ready-to-run algorithm examples
├── tests/
│   ├── unittests/          # Per-operation unit tests
│   ├── examples/           # End-to-end example tests
│   └── run_tests_distributed.py  # torchrun test launcher
├── docs/                   # Sphinx documentation
├── docker/                 # Docker build/run scripts
└── pyproject.toml          # Project metadata and tool config
```

## PR Guidelines

- Create a feature branch: `git checkout -b $USER/feature-name`
- Run linting and tests before opening a PR:

```bash
ruff check . --fix && ruff format .
python tests/run_tests_distributed.py tests/unittests/ --num_ranks 2 -v
```

- Add or update tests for any code you change.
- Update documentation under `docs/` for user-visible behavior changes.
- Fill in the PR description with a clear summary of what changed and why.
