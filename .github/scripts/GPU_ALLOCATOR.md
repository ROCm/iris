# GPU Allocator for CI

This directory contains a lightweight GPU allocator for CI jobs running on single-node multi-GPU machines.

## Overview

The GPU allocator provides deterministic GPU isolation for CI jobs without requiring heavy infrastructure like Slurm. It uses a simple file-based locking mechanism to ensure no overlapping GPU usage while maximizing machine utilization.

## Features

- **Variable GPU requests**: Support for requesting 1, 2, 4, or 8 GPUs
- **No overlapping usage**: Uses `flock` for atomic state management
- **High utilization**: Throughput-oriented scheduling (non-FIFO acceptable)
- **Automatic cleanup**: GPUs are released on job exit (via trap)
- **Lightweight**: No dependencies beyond bash and flock

## Design

### State Management

- **Lock file**: `/tmp/iris_gpu_state.lock` - Ensures atomic operations
- **State file**: `/tmp/iris_gpu_state` - Tracks available GPUs (comma-separated list)
- **Algorithm**: First-available scheduling

### Allocation Flow

1. Job requests N GPUs by calling `acquire_gpus N`
2. Script acquires exclusive lock
3. If enough GPUs available → allocate and update state
4. Otherwise → release lock, sleep, and retry
5. On allocation, `ALLOCATED_GPUS` environment variable is set
6. On job exit, GPUs are automatically released back to pool

## Usage

### In CI Scripts

```bash
# Source the allocator
source .github/scripts/gpu_allocator.sh

# Request GPUs (blocks until available)
acquire_gpus 4  # Request 4 GPUs

# Use allocated GPUs
echo "Using GPUs: $ALLOCATED_GPUS"
export HIP_VISIBLE_DEVICES=$ALLOCATED_GPUS

# Run your workload
./run_tests.sh

# GPUs are automatically released on exit (via trap)
# Or manually release:
release_gpus
```

### In GitHub Actions

The allocator is integrated into `run_tests.sh`. Simply omit the `gpu_devices` parameter:

```yaml
- name: Run tests
  run: |
    bash .github/scripts/run_tests.sh \
      "examples" \
      "4" \
      "" \
      "editable"
```

## Configuration

Environment variables (optional):

- `GPU_STATE_FILE`: Path to state file (default: `/tmp/iris_gpu_state`)
- `MAX_GPUS`: Total number of GPUs (default: `8`)
- `RETRY_DELAY`: Seconds between retries (default: `2`)
- `MAX_RETRIES`: Maximum retry attempts (default: `300`, ~10 minutes)

## Testing

Run the test suite:

```bash
bash tests/test_gpu_allocator.sh
```

This tests:
- State initialization
- Single and multi-GPU allocation
- GPU release and cleanup
- Allocation failures
- Sequential allocations (no overlap)

## Implementation Details

### Atomic Operations

All state modifications are protected by `flock` on the lock file:

```bash
(
    flock -x 200  # Exclusive lock
    # ... modify state ...
) 200>"$GPU_LOCK_FILE"
```

### Cleanup Handling

A trap ensures GPUs are always released:

```bash
trap cleanup_gpus EXIT
```

This prevents GPU leaks even if jobs crash or are killed.

### Throughput Optimization

The allocator is throughput-oriented, not FIFO:
- Jobs requesting fewer GPUs may complete before jobs requesting more
- This maximizes overall machine utilization
- Acceptable for CI workloads where individual job fairness is less critical

## Troubleshooting

### Jobs stuck waiting for GPUs

Check current state:
```bash
cat /tmp/iris_gpu_state
```

Should show comma-separated list of available GPUs (0-7 for 8-GPU system).

### Leaked GPUs (none available but no jobs running)

Reset state manually:
```bash
echo "0,1,2,3,4,5,6,7" > /tmp/iris_gpu_state
```

### Lock contention

If many jobs are waiting, consider:
- Reducing `MAX_RETRIES` to fail faster
- Increasing `RETRY_DELAY` to reduce lock contention
- Batching smaller jobs together

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.
