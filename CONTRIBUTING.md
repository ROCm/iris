# Contributing to Iris

Thank you for your interest in contributing to Iris! This document provides guidelines for contributing to the project.

> **Security vulnerabilities** — do not open a public GitHub issue. See [SECURITY.md](SECURITY.md) for the private reporting process.

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b $USER/your-feature-name
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run code quality checks
ruff check .
ruff format .

# Run tests 
python tests/run_tests_distributed.py tests/examples/test_all_load_bench.py --num_ranks 2 -v
python tests/run_tests_distributed.py tests/unittests/ --num_ranks 2 -v

# Or run individual test files
python tests/run_tests_distributed.py tests/examples/test_load_bench.py --num_ranks 2 -v
```

### 4. Commit and Push
```bash
git add .
git commit -m "Description of your changes"
git push origin $USER/your-feature-name
```

### 5. Create a Pull Request
- Go to the GitHub repository
- Create a new pull request from your branch
- Fill in the PR description with details about your changes
- Feel free to open a draft PR and ask for early feedback while you're still working on your changes

## Security Requirements

Contributors must not:

- Commit secrets, tokens, passwords, or credentials
- Introduce vulnerable dependencies without justification
- Bypass security controls or required security reviews

All contributions are subject to automated security scanning (secret scanning, code scanning,
dependency monitoring, vulnerability scanning) via the PR security scan workflow.

## License

By contributing to Iris, you agree that your contributions will be licensed under the MIT License.

## Governance

AMD employees must also follow the ROCm open source software contributing policies at
http://u.amd.com/rocm-oss-policies.

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.
