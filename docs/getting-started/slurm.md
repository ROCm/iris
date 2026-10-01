# Running Iris on SLURM

This guide covers a practical Iris workflow on SLURM-managed GPU clusters. It is written to stay generic across clusters while matching the provided Iris scripts and working well on clusters where:

- GPU nodes are scheduled with SLURM
- Docker is available on compute nodes, but not necessarily on login nodes
- fast local storage such as `/scratch` is preferred for builds and test output

## What the provided SLURM script assumes

The repository includes `scripts/run_core_tests_slurm.sh`, a batch wrapper for running `scripts/run_core_tests.sh`.

It **assumes the container image already exists**. It does **not** build `iris-dev` for you.

By default, the script:

- requests 1 node with 4 GPUs
- expects a Docker image named `iris-dev`
- stages the repository into node-local storage when available
- installs Iris in editable mode inside the container
- runs `scripts/run_core_tests.sh`
- copies the per-test logs back to `$HOME/slurm-logs/iris-core-tests-<jobid>/`

If the image is missing, the job fails fast with an explicit error.

## Fresh-clone workflow

### 1. Clone the repository on shared storage

Clone Iris somewhere visible from both the login node and the compute nodes.

```bash
git clone https://github.com/ROCm/iris.git
cd iris
```

If your cluster provides both shared storage and node-local scratch, keep the source tree on shared storage and let jobs copy into scratch for execution.

### 2. Request an interactive GPU allocation

If Docker is only available on worker nodes, first allocate a node and enter it.

```bash
salloc --nodes=1 --gres=gpu:4 --time=02:00:00
srun --pty $SHELL
```

Adjust GPUs, walltime, partition, account, memory, and CPU count to match your site policy.

### 3. Build the Iris Docker image on the allocated node

```bash
cd /path/to/iris
./docker/build.sh
```

This builds the default image name, `iris-dev`.

If you want a custom image name:

```bash
./docker/build.sh my-iris-image
```

You can verify that the image exists with:

```bash
docker image inspect iris-dev
```

### 4. Submit the batch job

From the repository root:

```bash
sbatch scripts/run_core_tests_slurm.sh
```

If you built a custom image:

```bash
sbatch --export=ALL,IMAGE_NAME=my-iris-image scripts/run_core_tests_slurm.sh
```

## Important note about node-local images

Some clusters store Docker images per node rather than in a shared registry-backed cache. In that setup, building `iris-dev` on one node does not guarantee that another node can see it.

If your cluster behaves this way, either:

1. build and submit on the same node, or
2. pin the batch job to the node where the image was built, or
3. rebuild the image on the target node

For example, after building the image on a worker node:

```bash
NODE_NAME=$(hostname)
sbatch -w "$NODE_NAME" scripts/run_core_tests_slurm.sh
```

If your cluster has shared container storage, you can usually omit `-w`.

## Monitoring the job

Use normal SLURM tools:

```bash
squeue -j <jobid>
sacct -j <jobid>
```

By default, the batch script writes SLURM stdout/stderr to:

```bash
iris_core_tests_<jobid>.out
```

in the directory where `sbatch` was invoked.

The per-test logs are copied to:

```bash
$HOME/slurm-logs/iris-core-tests-<jobid>/
```

## Running interactively inside the container

For development on an allocated node, you can also start the container manually:

```bash
./docker/run.sh iris-dev "$(pwd)"
```

Then install Iris in editable mode:

```bash
pip install -e ".[dev]"
```

This is useful when you want to debug failures before switching back to `sbatch`.

## Running example programs under SLURM

Many examples under `examples/` can be run directly with `python ... --num_ranks <N>` after Iris is installed in the container.

The repository includes a generic example wrapper:

```bash
scripts/run_example_slurm.sh
```

It stages the repository into node-local storage, installs Iris in the container, runs a chosen example script, and copies any `logs/` or `results/` directories back to:

```bash
$HOME/slurm-logs/iris-example-<jobid>/
```

### Generic usage

Submit any repo-relative example script and pass the example arguments after it:

```bash
sbatch scripts/run_example_slurm.sh <example_script> [example args...]
```

For example:

```bash
sbatch scripts/run_example_slurm.sh examples/00_load/load_bench.py --num_ranks 4
sbatch scripts/run_example_slurm.sh examples/13_flash_decode/example_run.py --num_ranks 4
```

### Example: `examples/14_all_gather_gemm`

This example directory provides both a pull-model and push-model entrypoint.

Pull model:

```bash
sbatch scripts/run_example_slurm.sh \
    examples/14_all_gather_gemm/example_run_pull.py \
    --num_ranks 4
```

Push model:

```bash
sbatch scripts/run_example_slurm.sh \
    examples/14_all_gather_gemm/example_run_push.py \
    --num_ranks 4
```

If your image is node-local, build on a worker node first and optionally pin the submission to that node:

```bash
NODE_NAME=$(hostname)
sbatch -w "$NODE_NAME" scripts/run_example_slurm.sh \
    examples/14_all_gather_gemm/example_run_pull.py \
    --num_ranks 4
```

Use a rank count that matches the GPUs allocated to the job.

### Custom image or install method

```bash
sbatch --export=ALL,IMAGE_NAME=my-iris-image scripts/run_example_slurm.sh \
    examples/14_all_gather_gemm/example_run_pull.py \
    --num_ranks 4
```

```bash
sbatch --export=ALL,INSTALL_METHOD=install scripts/run_example_slurm.sh \
    examples/14_all_gather_gemm/example_run_pull.py \
    --num_ranks 4
```

## Customizing the provided batch wrapper

The provided script is intentionally conservative and is meant for a 4-GPU core-test workflow.

Common customizations:

### Use a different image name

```bash
sbatch --export=ALL,IMAGE_NAME=my-iris-image scripts/run_core_tests_slurm.sh
```

### Store copied logs elsewhere

```bash
sbatch --export=ALL,PERSIST_LOG_ROOT=$HOME/my-iris-logs scripts/run_core_tests_slurm.sh
```

### Use a different scratch location

If your cluster does not use `/scratch`, point the job at another fast workspace:

```bash
sbatch --export=ALL,WORK_ROOT=/path/to/local/workdir scripts/run_core_tests_slurm.sh
```

### Change SLURM resources

Either edit the `#SBATCH` lines in `scripts/run_core_tests_slurm.sh`, or override them at submission time:

```bash
sbatch --gres=gpu:4 --time=04:00:00 --cpus-per-task=32 scripts/run_core_tests_slurm.sh
```

The current wrapper is designed around 4 GPUs. Since `scripts/run_core_tests.sh` includes 1, 2, 4, and 8-rank configurations, the wrapper automatically skips 8-rank cases when only 4 GPUs are visible.

## Troubleshooting

### `Docker image iris-dev not found`

Build the image first:

```bash
./docker/build.sh
```

If the image was built on another worker node, submit to that same node or rebuild locally.

### `docker` is not available on the login node

Request an interactive allocation and build from inside the worker node:

```bash
salloc --nodes=1 --gres=gpu:4 --time=02:00:00
srun --pty $SHELL
./docker/build.sh
```

### The job should run from fast local storage

The provided wrapper already stages the repository into node-local storage when possible. If your cluster uses a different path than `/scratch`, set `WORK_ROOT` when submitting.

### I need an Apptainer-based workflow instead

Iris also includes Apptainer support:

```bash
./apptainer/build.sh
./apptainer/run.sh
```

The provided `scripts/run_core_tests_slurm.sh` wrapper is Docker-based, so use the Apptainer scripts directly or create a cluster-specific batch wrapper around them.
