#!/bin/bash

NUM_GPUS=2
OMP_NUM_THREADS=1
COLLECTIVE="all_gather"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --num-gpus NUM      Number of GPUs to use (default: $NUM_GPUS)"
    echo "  -c, --collective NAME   Collective operation to run (default: $COLLECTIVE)"
    echo "  -h, --help              Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 -n 4 -c all_reduce  # Run with 4 GPUs using all_reduce"
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num-gpus) NUM_GPUS="$2"; shift ;;
        -c|--collective) COLLECTIVE="$2"; shift ;;
        -h|--help) show_help ;;
        *) echo "Unknown parameter: $1"; show_help ;;
    esac
    shift
done

export OMP_NUM_THREADS=$OMP_NUM_THREADS

echo "Running '$COLLECTIVE' with $NUM_GPUS GPUs..."
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} "${COLLECTIVE}.py"