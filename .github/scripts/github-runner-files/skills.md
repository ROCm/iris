# Build instructions

## Container build (SLURM)

From the `github-runner` directory:

```bash
sbatch build-github-coding-agent-runner.sh
```

- **Partition:** `mi3001x`
- **Time limit:** 2 hours
- **Input:** definition file, default `iris.def` (override with `--def=FILE` or env `DEF_FILE`)
- **Output:** default `github-copilot-coding-agent-runner.sif` (override with `--output=FILE` or env `OUTPUT_SIF`)

The job uses `SLURM_SUBMIT_DIR` when set, so submit from the repo directory (e.g. `cd /path/to/github-runner && sbatch build-github-coding-agent-runner.sh`) so the build runs in the right place.

Temp and cache are under the build directory (`.apptainer-tmp`, `.apptainer-cache`) to avoid filling `/tmp`. The temp dir is removed after a successful build; the cache is kept for faster rebuilds. To reclaim space, remove `.apptainer-cache` as well.

## After build

**Option 1 — Run via SLURM with env (SLURM-only fallback):** set `GITHUB_TOKEN` and `GITHUB_REPOSITORY`, then submit the script. The script uses `SLURM_SUBMIT_DIR` and `WORK` (when set) for script-dir and runner-base.

```bash
export GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo
sbatch run-github-coding-agent-runner.sh
```

**Option 2 — Run standalone with flags:** pass all required options on the command line (see README).

See **README.md** for full setup and usage.
