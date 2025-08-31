# Contributing to Iris

Thank you for your interest in contributing to Iris! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- **AMD GPU**: MI300X, MI350X, MI355X, or other ROCm-compatible GPUs
- **ROCm**: AMD's GPU compute platform (version 5.7+ recommended)
- **Python**: Python 3.8 or higher
- **MPI**: OpenMPI or MPICH for multi-GPU communication
- **Git**: For version control

### Containerized Development

Iris provides containerized development environments:

#### Using Docker
```bash
cd docker
./build.sh iris-dev
./run.sh iris-dev
pip install -e ".[dev]"
```

#### Using Apptainer
```bash
cd apptainer
./build.sh
./run.sh
pip install -e ".[dev]"
```

### Manual Setup
```bash
git clone https://github.com/ROCm/iris.git
cd iris
python3 -m venv iris-env
source iris-env/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

### 1. Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the upstream remote

### 2. Create a Feature Branch
```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

**Branch Naming:**
- `feature/descriptive-name`: New features
- `bugfix/issue-description`: Bug fixes
- `docs/documentation-update`: Documentation changes

### 3. Code Style Guidelines

- **Python Code**: Follow PEP 8
- **Line Length**: Maximum 120 characters
- **Documentation**: Use Google-style docstrings
- **Type Hints**: Include type hints for public functions

### 4. Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unittests/test_atomic_add.py

# Test examples
mpirun -np 2 python examples/00_load/load_bench.py
mpirun -np 4 python examples/04_atomic_add/atomic_add_bench.py
```

### 5. Commit Guidelines
Use conventional commit format:
```bash
git commit -m "feat(atomic): add atomic_min operation support

- Implement atomic_min for int32 and float32 types
- Add comprehensive tests for edge cases
- Update documentation with usage examples

Closes #123"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes

### 6. Pull Request
```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

## Areas for Contribution

### High Priority
- **Performance optimization**: Improve GEMM and communication performance
- **GPU support**: Extend support to more AMD GPU models
- **Documentation**: Improve tutorials and API documentation
- **Testing**: Add more comprehensive test coverage

### Medium Priority
- **New communication patterns**: Implement additional collective operations
- **Error handling**: Improve error messages and debugging
- **Examples**: Add more real-world use case examples

## Getting Help

### Before Asking
1. Check existing GitHub issues
2. Read relevant documentation
3. Try existing examples
4. Search GitHub Discussions

### Communication Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

## Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome contributors from all backgrounds
- **Be constructive**: Provide constructive feedback
- **Be patient**: Understand that contributors have different experience levels

## License

By contributing to Iris, you agree that your contributions will be licensed under the MIT License.

## Getting Started

Ready to contribute? Start small:
1. Fix a typo or add a simple test
2. Study existing examples and tests
3. Join discussions on GitHub
4. Submit focused, small changes

---

**Thank you for contributing to Iris!**
