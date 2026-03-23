#!/bin/bash

# Cleanup script for old GitHub runner configurations and overlays

set -e

WORK_DIR="${WORK:-/work1/amd/josantos}"
RUNNER_BASE="${WORK_DIR}/github-runner-data"
OVERLAY_DIR="${RUNNER_BASE}/overlays"

echo "=========================================="
echo "GitHub Runner Cleanup Script"
echo "=========================================="
echo "Cleaning up directories in: ${RUNNER_BASE}"
echo ""

# Function to check if SLURM job is still running
is_job_running() {
    local job_id=$1
    squeue -j "$job_id" &>/dev/null
}

# Cleanup old runner config directories
echo "Cleaning up old runner configurations..."
for runner_dir in "${RUNNER_BASE}"/.github-runner-*; do
    if [ -d "$runner_dir" ]; then
        # Extract job ID from directory name
        job_id=$(basename "$runner_dir" | sed 's/.github-runner-//')
        
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            # Check if job is still running
            if is_job_running "$job_id"; then
                echo "  Skipping $runner_dir (job $job_id is still running)"
            else
                echo "  Removing $runner_dir (job $job_id is not running)"
                rm -rf "$runner_dir"
            fi
        else
            echo "  Skipping $runner_dir (not a job-specific directory)"
        fi
    fi
done

# Cleanup old overlay images
echo "Cleaning up old overlay images..."
for overlay_file in "${OVERLAY_DIR}"/overlay-*.img; do
    if [ -f "$overlay_file" ]; then
        # Extract job ID from filename
        job_id=$(basename "$overlay_file" | sed 's/overlay-//' | sed 's/.img$//')
        
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            # Check if job is still running
            if is_job_running "$job_id"; then
                echo "  Skipping $overlay_file (job $job_id is still running)"
            else
                size=$(du -h "$overlay_file" | cut -f1)
                echo "  Removing $overlay_file (job $job_id is not running, size: $size)"
                rm -f "$overlay_file"
            fi
        else
            echo "  Skipping $overlay_file (not a job-specific overlay)"
        fi
    fi
done

echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="

# Show remaining files
echo "Remaining runner configurations:"
ls -lh "${RUNNER_BASE}"/.github-runner-* 2>/dev/null || echo "  None"

echo ""
echo "Remaining overlay images:"
ls -lh "${OVERLAY_DIR}"/overlay-*.img 2>/dev/null || echo "  None"

# Show disk usage
echo ""
echo "Disk usage:"
echo "  Runner data directory: $(du -sh "${RUNNER_BASE}" 2>/dev/null | cut -f1)"
echo "  Overlays directory: $(du -sh "${OVERLAY_DIR}" 2>/dev/null | cut -f1)"

