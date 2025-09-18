# Contributing to Iris

Thank you for your interest in contributing to Iris! This document provides guidelines for contributing to the project.

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

## License

By contributing to Iris, you agree that your contributions will be licensed under the MIT License.
