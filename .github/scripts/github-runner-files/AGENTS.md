# Agent instructions

## Workflow

Flow: **run-github-coding-agent-runner.sh** → **container** → **start.sh** → **Actions listener**.

- **Standalone:** run `./run-github-coding-agent-runner.sh` with required flags (`--github-token`, `--github-repository`, `--script-dir`, `--runner-base`). No env needed.
- **SLURM:** set `GITHUB_TOKEN` and `GITHUB_REPOSITORY`, then `sbatch run-github-coding-agent-runner.sh`. When the script runs under SLURM with no arguments, it uses env and SLURM defaults (`SLURM_SUBMIT_DIR`, `WORK`) for script-dir and runner-base. start.sh installs/configures the runner in `RUNNER_HOME` if needed and starts the Actions listener; workflow jobs run in the container.

## Conventions

When editing scripts or config in this project:

1. **Never add sensitive data to scripts or committed files.**  
   Do not hardcode tokens, passwords, API keys, or other secrets. Use environment variables or a secure mechanism outside the repo (e.g. `export GITHUB_TOKEN` before running).

2. **Never use host-specific absolute paths.**  
   Do not add paths like `/work1/amd/josantos/...` or other machine-specific directories. Prefer:
   - Paths relative to the script (e.g. `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` then `cd "${SCRIPT_DIR}"`).
   - Environment variables (e.g. `$WORK`, `$HOME`) when a base directory is needed.
   - Relative paths from the project or script location.

3. **Never edit the container definition file (e.g. `iris.def`) unless explicitly asked.**  
   Prefer changing scripts (e.g. `start.sh`, `run-github-coding-agent-runner.sh`) to install, configure, or run things at runtime. Only modify `.def` files when the user explicitly requests it.
