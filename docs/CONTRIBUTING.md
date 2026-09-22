# Contributing to Iris

The contribution guidelines live at the repository root:
[CONTRIBUTING.md](https://github.com/ROCm/iris/blob/main/CONTRIBUTING.md).

They cover the development workflow, code style, testing requirements, security
requirements, and governance.

> **Security vulnerabilities** — do not open a public GitHub issue. See
> [SECURITY.md](https://github.com/ROCm/iris/blob/main/SECURITY.md) for the private
> reporting process.

## Quick reference

Create a feature branch:

```bash
git checkout -b $USER/your-feature-name
```

Run code quality checks and tests before pushing:

```bash
ruff check .
ruff format .

python tests/run_tests_distributed.py tests/unittests/ --num_ranks 2 -v
```
