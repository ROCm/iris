# Contributing to Iris

Thank you for your interest in contributing to Iris! This document provides comprehensive guidelines for contributing to the project, from development setup to submitting pull requests.

## Why Contribute?

Iris is an open-source project that benefits from community contributions. By contributing, you can:

- **Improve the framework**: Add new features, fix bugs, and optimize performance
- **Learn from experts**: Work with GPU and distributed computing specialists
- **Build your portfolio**: Showcase your skills in cutting-edge GPU programming
- **Help the community**: Make multi-GPU programming accessible to more developers

## Development Setup

### Prerequisites

Before contributing to Iris, ensure you have:

- **AMD GPU**: MI300X, MI350X, MI355X, or other ROCm-compatible GPUs
- **ROCm**: AMD's GPU compute platform (version 5.7+ recommended)
- **Python**: Python 3.8 or higher
- **MPI**: OpenMPI or MPICH for multi-GPU communication
- **Git**: For version control

### Containerized Development (Recommended)

Iris provides containerized development environments for consistent development:

#### Using Docker

```bash
# Build the development image
cd docker
./build.sh iris-dev

# Run the container
./run.sh iris-dev

# Install Iris in development mode with dev dependencies
pip install -e ".[dev]"
```

#### Using Apptainer

For HPC environments or systems where Docker is not available:

```bash
# Build the Apptainer image
cd apptainer
./build.sh

# Run the container
./run.sh

# Install Iris in development mode
pip install -e ".[dev]"
```

### Manual Development Setup

For advanced users who want full control:

```bash
# Clone the repository
git clone https://github.com/ROCm/iris.git
cd iris

# Create virtual environment
python3 -m venv iris-env
source iris-env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Development Workflow

### 1. Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/iris.git
   cd iris
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ROCm/iris.git
   ```

### 2. Create a Feature Branch

Always work on a feature branch, never on main:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create and switch to feature branch
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions:**
- `feature/descriptive-name`: New features
- `bugfix/issue-description`: Bug fixes
- `docs/documentation-update`: Documentation changes
- `test/test-addition`: Test additions
- `refactor/code-improvement`: Code refactoring

### 3. Make Your Changes

#### Code Style Guidelines

Iris follows strict code quality standards:

1. **Python Code**: Follow PEP 8 with modifications
2. **Line Length**: Maximum 120 characters
3. **Imports**: Group imports (standard library, third-party, local)
4. **Documentation**: Use Google-style docstrings
5. **Type Hints**: Include type hints for all public functions

#### Example Code Style

```python
def compute_gemm(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    result: torch.Tensor,
    block_size: int = 32
) -> None:
    """Compute matrix multiplication C = A @ B.
    
    Args:
        matrix_a: Input matrix A of shape (M, K)
        matrix_b: Input matrix B of shape (K, N)
        result: Output matrix C of shape (M, N)
        block_size: Block size for tiling. Default: 32
        
    Raises:
        ValueError: If matrix dimensions are incompatible
        RuntimeError: If computation fails
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    # Implementation here
    pass
```

#### Documentation Standards

1. **Docstrings**: Every public function must have a docstring
2. **Examples**: Include usage examples in docstrings
3. **Type Information**: Document parameter types and return values
4. **Error Conditions**: Document when exceptions are raised

### 4. Testing Your Changes

#### Run Code Quality Checks

```bash
# Check code style and quality
ruff check .

# Format code automatically
ruff format .

# Type checking (if mypy is configured)
mypy iris/
```

#### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unittests/test_atomic_add.py

# Run tests with coverage
pytest --cov=iris

# Run tests in parallel
pytest -n auto
```

#### Test Your Examples

```bash
# Test basic operations
mpirun -np 2 python examples/00_load/load_bench.py

# Test atomic operations
mpirun -np 4 python examples/04_atomic_add/atomic_add_bench.py

# Test GEMM examples
mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py --validate
```

### 5. Commit Your Changes

#### Commit Message Guidelines

Use conventional commit format:

```bash
# Format: type(scope): description
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
- `chore`: Maintenance tasks

#### Commit Best Practices

1. **Atomic commits**: Each commit should represent one logical change
2. **Descriptive messages**: Explain what and why, not how
3. **Reference issues**: Link commits to GitHub issues
4. **Separate concerns**: Don't mix different types of changes

### 6. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Pull Request Guidelines

### PR Description Template

```markdown
## Description

Brief description of what this PR accomplishes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated existing tests if needed
- [ ] Examples run successfully

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Examples tested
- [ ] No breaking changes (or documented if breaking)

## Related Issues

Closes #123
Related to #456
```

### PR Review Process

1. **Self-review**: Review your own changes before submitting
2. **CI checks**: Ensure all automated checks pass
3. **Code review**: Address feedback from maintainers
4. **Final approval**: Get approval from at least one maintainer

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
- **Benchmarks**: Create performance benchmarking suite

### Low Priority

- **Code cleanup**: Refactor and improve existing code
- **Documentation**: Fix typos and improve clarity
- **CI/CD**: Improve build and test automation

## Getting Help

### Before Asking

1. **Check existing issues**: Search GitHub issues for similar problems
2. **Read documentation**: Review relevant documentation sections
3. **Try examples**: Run existing examples to understand patterns
4. **Search discussions**: Check GitHub Discussions for similar questions

### Asking for Help

When asking for help, provide:

1. **Clear description**: What you're trying to accomplish
2. **Error messages**: Complete error output
3. **Environment details**: OS, Python version, GPU model
4. **Minimal example**: Smallest code that reproduces the issue
5. **What you've tried**: Steps you've already attempted

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions and improvements

## Recognition

### Contributors

All contributors are recognized in:

- **GitHub Contributors**: Automatic recognition for commits
- **Release Notes**: Credit for significant contributions
- **Documentation**: Attribution in relevant sections

### Significant Contributions

Contributors who make substantial contributions may be:

- **Added to README**: Listed as contributors
- **Given maintainer access**: For trusted contributors
- **Invited to meetings**: Participate in project discussions

## Code of Conduct

### Our Standards

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome contributors from all backgrounds
- **Be constructive**: Provide constructive feedback
- **Be patient**: Understand that contributors have different experience levels

### Unacceptable Behavior

- **Harassment**: Any form of harassment or discrimination
- **Trolling**: Deliberately disruptive behavior
- **Spam**: Unwanted promotional content
- **Personal attacks**: Attacking individuals rather than ideas

## License

By contributing to Iris, you agree that your contributions will be licensed under the MIT License. This means:

- **Your contributions**: Become part of the open-source project
- **Others can use**: Your code in their own projects
- **Attribution**: You'll be credited for your contributions
- **No warranty**: Code is provided "as is" without warranty

## Getting Started

Ready to contribute? Here's a quick path:

1. **Start small**: Fix a typo or add a simple test
2. **Learn the codebase**: Study existing examples and tests
3. **Join discussions**: Participate in GitHub Discussions
4. **Ask questions**: Don't hesitate to ask for clarification
5. **Submit PRs**: Start with small, focused changes

## Questions?

If you have questions about contributing:

- **Open a discussion**: Use GitHub Discussions for general questions
- **Create an issue**: Use GitHub Issues for specific problems
- **Contact maintainers**: Reach out to the development team

---

**Thank you for contributing to Iris! Your contributions help make multi-GPU programming accessible to everyone.**
