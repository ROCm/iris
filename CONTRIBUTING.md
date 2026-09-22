# Contributing to Iris

Full contribution guidelines live in [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md), covering the
development workflow, code style, testing requirements, and security requirements.

> **Security vulnerabilities** — do not open a public GitHub issue. See [SECURITY.md](SECURITY.md)
> for the private reporting process.

## Quick reference

```bash
# Feature branch
git checkout -b $USER/your-feature-name

# Code quality
ruff check .
ruff format .

# Tests
python tests/run_tests_distributed.py tests/unittests/ --num_ranks 2 -v
```

AMD employees must also follow the ROCm open source software contributing policies at
http://u.amd.com/rocm-oss-policies.

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.
