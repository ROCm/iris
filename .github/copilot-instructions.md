# Iris: Multi-GPU Programming Framework

## Description

Iris is a Triton-based framework for Remote Memory Access (RMA) operations on AMD GPUs. It provides SHMEM-like APIs within Triton for Multi-GPU programming with:

- Clean abstractions with full symmetric heap implementation
- Pythonic PyTorch-like host APIs for tensor operations  
- Triton-style device APIs for load, store, and atomic operations
- Minimal dependencies (Triton, PyTorch, HIP runtime, mpi4py)
- Comprehensive examples showing communication/computation overlap

**FOLLOW THESE INSTRUCTIONS EXACTLY. Reference these instructions first before using search or bash commands.**

## Prerequisites

- **GPU**: AMD GPUs with ROCm compatibility (tested on MI300X, MI350X & MI355X)
  > **Note**: See below for instructions on development without AMD GPU access
- **ROCm/HIP Toolkit**: Required for building C++/HIP components
- **MPI**: Required for multi-GPU operations
- **Docker/Apptainer**: Recommended for containerized development

## Build

### Docker Development Environment (Recommended)
```bash
# Build and start development container (takes 45-60 minutes - NEVER CANCEL)
docker compose up --build -d

# Attach to running container
docker attach iris-dev

# Install Iris in development mode
cd iris && pip install -e ".[dev]"
```

### Alternative Docker Setup
```bash
# Build Docker image manually
./docker/build.sh <image-name>  # Takes 45-60 minutes

# Run container
./docker/run.sh <image-name>

# Install Iris
cd iris && pip install -e ".[dev]"
```

### Apptainer Setup
```bash
# Build and run Apptainer image
./apptainer/build.sh
./apptainer/run.sh

# Install Iris
pip install -e ".[dev]"
```

### Local Development (Not Recommended)
```bash
# Requires ROCm/HIP toolkit installation
pip install -e ".[dev]"
```

### Development Without AMD GPU
If you don't have access to AMD GPUs, you can still contribute to the project:
- **Code Editing**: Start editing code directly in your local environment
- **CI Testing**: The project has comprehensive CI pipelines that will test your changes automatically. You can check the CI logs if your changes fail to understand what went wrong.
- **Local Validation**: Run linting and formatting locally: `ruff check . --fix && ruff format .`

## Run

### Testing
```bash
# Run unit tests
pytest tests/unittests/

# Run example tests  
pytest tests/examples/

# Run specific example (requires MPI and GPU)
mpirun -np 8 python examples/00_load/load_bench.py
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
