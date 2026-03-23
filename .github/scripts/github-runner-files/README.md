# Iris + GitHub Actions Self-Hosted Runner (Apptainer)

This setup runs a GitHub Actions self-hosted runner in an Apptainer container with the Iris framework (ROCm, Triton) and the `copilot` label, for HPC environments where Docker is not available.

## Prerequisites

- Apptainer/Singularity installed
- GitHub Personal Access Token with `repo` scope
- Access to the repository where you want to register the runner
- SLURM (for job scheduling)
- Optional: ROCm/AMD GPU partition for GPU workflows

## Quick Start

### 1. Create GitHub Personal Access Token

1. Go to https://github.com/settings/tokens/new
2. Name: e.g. `GitHub Actions Runner`
3. Scopes: Select `repo` (Full control of private repositories)
4. Click "Generate token" and save it securely

### 2. Prepare token and paths

You will pass the GitHub token and repository as flags (see step 4). Do not commit tokens.

### 3. Build the Container

From this directory:

```bash
sbatch build-github-coding-agent-runner.sh
```

This builds `github-copilot-coding-agent-runner.sif` from `iris.def` by default. To use another definition file: `./build-github-coding-agent-runner.sh --def=my.def` or set `DEF_FILE=my.def` before `sbatch`. The job uses partition `mi3001x` and may take a while. See **skills.md** for full build instructions.

### 4. Run the Runner

After the build completes, from the repo directory (where `run-github-coding-agent-runner.sh` and the `.sif` live). You can run in two ways:

**Option A — Standalone with flags (required when not using SLURM):**

```bash
./run-github-coding-agent-runner.sh \
  --github-token='YOUR_GITHUB_TOKEN' \
  --github-repository='owner/repo' \
  --script-dir="$(pwd)" \
  --runner-base="$(pwd)/runner-data"
```

**Option B — Via SLURM with environment variables (when `sbatch run-github-coding-agent-runner.sh` is used, the script uses env and SLURM defaults for any value not passed as a flag):**

```bash
export GITHUB_TOKEN='YOUR_GITHUB_TOKEN'
export GITHUB_REPOSITORY='owner/repo'
sbatch run-github-coding-agent-runner.sh
```

With Option B, `SCRIPT_DIR` defaults to `SLURM_SUBMIT_DIR` (or the script’s directory), and `RUNNER_BASE` defaults to `$WORK/github-runner-data` if `WORK` is set, otherwise `$SCRIPT_DIR/github-runner-data`. You can override with `export SCRIPT_DIR=... RUNNER_BASE=...` if needed.

Copy-paste and replace:
- `YOUR_GITHUB_TOKEN` — your GitHub Personal Access Token
- `owner/repo` — your repository (e.g. `Jose/Iris`)
- `runner-data` (Option A) — directory for runner state and work (created if missing); use any path you prefer.

Optional flags (Option A) or env vars (Option B) (examples):

```bash
  --cluster-name='vultr-k8' \   # or export CLUSTER_NAME=...
  --runner-labels='copilot,rocm' \
  --use-overlay=1
```

### 5. Verify Runner Registration

1. Go to your repository on GitHub
2. Navigate to: Settings → Actions → Runners
3. You should see your runner listed with the `copilot` label

## Using the Runner in Workflows

In your `.github/workflows/*.yml` files, use the runner via the `copilot` label (or whatever you passed to `--runner-labels`). Ensure the workflow’s `runs-on` matches: e.g. `runs-on: copilot` or `runs-on: [self-hosted, copilot]`. If a workflow uses a different label (e.g. `apptainer`), either register the runner with that label too or change the workflow to `copilot`.

```yaml
name: Example Workflow
on: [push]

jobs:
  build:
    runs-on: copilot
    steps:
      - uses: actions/checkout@v4
      - name: Run a test
        run: echo "Running on Iris + copilot runner in HPC!"
```

## Workflow

End-to-end flow when you run the runner via SLURM:

1. **One-time setup**  
   Create a GitHub PAT with `repo` scope. From this directory, run `sbatch build-github-coding-agent-runner.sh` to build `github-copilot-coding-agent-runner.sif` from `iris.def` (Iris + ROCm; the runner is not in the image).

2. **Run the runner**  
   Either pass required flags to `run-github-coding-agent-runner.sh` (standalone) or set `GITHUB_TOKEN` and `GITHUB_REPOSITORY` and run `sbatch run-github-coding-agent-runner.sh` (SLURM-only env fallback; see step 4). The script runs Apptainer with overlay and bind mounts and executes `/bin/bash -c "/runner-scripts/start.sh"`. So: **run-github-coding-agent-runner.sh** → **container** → **start.sh**.

3. **Inside the container: start.sh**  
   It receives `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `RUNNER_HOME`, `RUNNER_NAME`, `RUNNER_LABELS`, and `RUNNER_WORKDIR` from the run script (via `--env`). It checks required vars, sets defaults for any unset, and uses `RUNNER_HOME` (e.g. `/runner-home`). If the runner is not installed in `RUNNER_HOME`, it installs it (from `/opt/actions-runner` or by download). It fetches a registration token from GitHub, runs `config.sh`, then starts the Actions runner listener (`./run.sh`). The runner listens for jobs; when a workflow uses the `copilot` (or your) label, GitHub sends a job and the runner runs the steps in the container.

4. **End-to-end**  
   You run **run-github-coding-agent-runner.sh** with `--github-token`, `--github-repository`, `--script-dir`, and `--runner-base` (and optionally `--sif`). **run-github-coding-agent-runner.sh** starts the container, binds the script dir and runner dirs, passes env to the container, and runs **start.sh**. **start.sh** installs/configures the runner if needed and starts the listener. So: **run-github-coding-agent-runner.sh** → **container** → **start.sh** (install/configure + listener) → **runner runs workflow jobs**.

## Management Commands

```bash
# Build container
sbatch build-github-coding-agent-runner.sh

# Run standalone (required flags)
./run-github-coding-agent-runner.sh --github-token='...' --github-repository='owner/repo' --script-dir="$(pwd)" --runner-base="$(pwd)/runner-data"

# Run via SLURM with env (set GITHUB_TOKEN and GITHUB_REPOSITORY; SCRIPT_DIR/RUNNER_BASE default from SLURM)
export GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo
sbatch run-github-coding-agent-runner.sh

# Check SLURM job status
squeue -u $USER

# View SLURM job logs
tail -f github-coding-agent-runner-*.out

# Cancel SLURM job
scancel <job_id>
```

## Customization

### Runner Name and Labels

Defaults are set in `run-github-coding-agent-runner.sh` (e.g. runner name: `repo-runner-cluster-YYYYMMDD-HHMMSS`; default label: `copilot`). Override with flags:

```bash
./run-github-coding-agent-runner.sh ... --runner-name='my-runner' --runner-labels='copilot,slurm,apptainer,hpc,iris,rocm,mi300x'
```

### SLURM Parameters

Edit `run-github-coding-agent-runner.sh` SBATCH directives as needed:

- `#SBATCH --time=8:00:00`
- `#SBATCH -p mi3008x`  # partition
- `#SBATCH --nodes=1`

GPU access is enabled via `--rocm` in the container run.

### Kubernetes / no overlay

Overlays are not used in Kubernetes (default `USE_OVERLAY=0` in pods). The script uses **bind mounts only** for writable space:

- **RUNNER_HOME** (runner config) and **RUNNER_WORKDIR** (job work) are bind-mounted from the host/pod.
- Optional: set **RUNNER_TMP** to a writable directory (e.g. a pod `emptyDir` mounted in the container) and the script will bind it to `/tmp` inside the container so tools (e.g. Triton cache) can write there.

Example in a pod spec: mount an `emptyDir` at `/runner-tmp` and set `RUNNER_TMP=/runner-tmp` in the container env so `/tmp` is writable without an overlay.

## Troubleshooting

### Runner not appearing in GitHub

1. Check logs: `tail -f github-coding-agent-runner-*.out` and `github-coding-agent-runner-*.err`
2. Verify the token (`--github-token` or `GITHUB_TOKEN`) has `repo` scope
3. Verify `--github-repository` format is `owner/repo`
4. Check token has not expired

### Build failures

- Build runs on partition `mi3001x` with fakeroot. See **skills.md** for details.
- Cache and temp dirs are under the project directory (`.apptainer-cache`, `.apptainer-tmp`). Ensure enough disk space.

### Container not found when running

If the container image is missing (default: `script-dir/github-copilot-coding-agent-runner.sif`), `run-github-coding-agent-runner.sh` will print a message. Run the build and wait for it to complete, or pass `--sif=/path/to/image.sif`.

### Runner offline

```bash
squeue -u $USER
tail -50 github-coding-agent-runner-*.err
scancel <job_id>
# Resubmit: either same flags (standalone) or same env then sbatch run-github-coding-agent-runner.sh
```

## Security

- **Tokens**: Never commit tokens. Use `--github-token=TOKEN` when running standalone, or set `GITHUB_TOKEN` when using `sbatch run-github-coding-agent-runner.sh`; do not put secrets in committed files.
- **Paths**: Do not hardcode host-specific paths in scripts. See **AGENTS.md** for project conventions.
- **Container**: Apptainer runs as your user; the container is read-only with a per-job writable overlay.

## File Structure

```
github-runner/
├── iris.def                        # Apptainer definition (Iris + ROCm)
├── build-github-coding-agent-runner.sh   # SLURM build job (--def=FILE for definition file)
├── run-github-coding-agent-runner.sh   # Run job (flags or sbatch + env)
├── start.sh                             # Runner startup (inside container; also used as K8s entrypoint)
├── runner-container.env.example    # Example env file for container (start.sh sources it)
├── AGENTS.md                       # Agent instructions (no secrets, relative paths)
├── skills.md                       # Build instructions
├── README.md                       # This file
└── github-copilot-coding-agent-runner.sif   # Built image (after build)
```

## License

MIT License.
