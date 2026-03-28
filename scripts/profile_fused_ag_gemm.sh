#!/bin/bash
# SPDX-License-Identifier: MIT
# Profile fused AG+GEMM with rocprofv3
#
# Usage:
#   bash scripts/profile_fused_ag_gemm.sh [mode]
#
# Modes:
#   counters  - Hardware performance counters (default)
#   pc        - PC sampling (instruction-level hotspots)
#   att       - Address Translation Tracing (memory access patterns)
#   all       - Run all profiling modes
#
# Requires: rocprofv3 installed (typically at /opt/rocm/bin/rocprofv3)

set -euo pipefail

MODE="${1:-counters}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${REPO_DIR}/profiles/fused_ag_gemm"
NPROC="${NPROC:-4}"

mkdir -p "$OUTPUT_DIR"

# Detect rocprofv3
ROCPROFV3=""
for candidate in /opt/rocm/bin/rocprofv3 rocprofv3; do
    if command -v "$candidate" &>/dev/null; then
        ROCPROFV3="$candidate"
        break
    fi
done

if [ -z "$ROCPROFV3" ]; then
    echo "ERROR: rocprofv3 not found. Install ROCm profiling tools."
    exit 1
fi

echo "Using rocprofv3: $ROCPROFV3"
echo "Output directory: $OUTPUT_DIR"
echo "Mode: $MODE"
echo ""

# Create the benchmark script that rocprofv3 will profile
cat > /tmp/profile_ag_gemm.py << 'PYEOF'
import gc, os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.environ.get("IRIS_REPO", "."))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

import iris
from iris.ccl import Config

rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
total_sms = torch.cuda.get_device_properties(device).multi_processor_count

M, K_local, N = 2048, 2048, 4096
K = K_local * world_size
dtype = torch.float16

shmem = iris.iris(2**33)

torch.manual_seed(42 + rank)
A_shard = torch.randn(M, K_local, dtype=dtype, device=device)
torch.manual_seed(123)
weight = torch.randn(K, N, dtype=dtype, device=device)

A_sym = shmem.zeros((M, K_local), dtype=dtype)
A_sym.copy_(A_shard)
W_sym = shmem.zeros((K, N), dtype=dtype)
W_sym.copy_(weight)
output = shmem.zeros((M, N), dtype=dtype)
shmem.barrier()

config = Config(
    block_size_m=256, block_size_n=64,
    swizzle_size=6, comm_sms=total_sms,
    num_warps=4, num_stages=1,
)

# Warmup
for _ in range(20):
    shmem.ccl.all_gather_gemm(output, A_sym, W_sym, config=config, block_size_k=64)
torch.cuda.synchronize()

# Profiled iterations
for _ in range(10):
    shmem.ccl.all_gather_gemm(output, A_sym, W_sym, config=config, block_size_k=64)
torch.cuda.synchronize()

if rank == 0:
    print("Profile run complete")

shmem.barrier()
del shmem
gc.collect()
dist.destroy_process_group()
PYEOF

run_counters() {
    echo "=== Hardware Performance Counters ==="
    # Create counter input file
    cat > /tmp/rocprof_counters.txt << 'EOF'
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SMEM SQ_INSTS_LDS
pmc: SQ_WAIT_INST_VMEM SQ_WAIT_INST_LDS SQ_ACTIVE_INST_VALU
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum
pmc: TCP_TOTAL_READ_sum TCP_TOTAL_WRITE_sum TCP_TOTAL_CACHE_ACCESSES_sum
EOF

    IRIS_REPO="$REPO_DIR" $ROCPROFV3 \
        -i /tmp/rocprof_counters.txt \
        -o "$OUTPUT_DIR/counters" \
        -- python3 -m torch.distributed.run --nproc_per_node=$NPROC /tmp/profile_ag_gemm.py

    echo "Counter results in: $OUTPUT_DIR/counters/"
}

run_pc_sampling() {
    echo "=== PC Sampling ==="

    IRIS_REPO="$REPO_DIR" $ROCPROFV3 \
        --plugin pcsamp \
        -o "$OUTPUT_DIR/pcsamp" \
        -- python3 -m torch.distributed.run --nproc_per_node=$NPROC /tmp/profile_ag_gemm.py

    echo "PC sampling results in: $OUTPUT_DIR/pcsamp/"
}

run_att() {
    echo "=== Address Translation Tracing ==="

    IRIS_REPO="$REPO_DIR" $ROCPROFV3 \
        --plugin att \
        -o "$OUTPUT_DIR/att" \
        -- python3 -m torch.distributed.run --nproc_per_node=$NPROC /tmp/profile_ag_gemm.py

    echo "ATT results in: $OUTPUT_DIR/att/"
}

case "$MODE" in
    counters)
        run_counters
        ;;
    pc)
        run_pc_sampling
        ;;
    att)
        run_att
        ;;
    all)
        run_counters
        echo ""
        run_pc_sampling
        echo ""
        run_att
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: counters, pc, att, all"
        exit 1
        ;;
esac

echo ""
echo "Done. Profile outputs in: $OUTPUT_DIR/"
