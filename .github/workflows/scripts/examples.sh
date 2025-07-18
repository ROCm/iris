#!/bin/bash
set -e

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Run examples and store outputs
echo 'Running Iris examples...'

mkdir -p /iris_results

# Examples
mpirun -np 8 python examples/00_load/load_bench.py -o /iris_results/load_bench.json
mpirun -np 8 python examples/01_store/store_bench.py -o /iris_results/store_bench.json


mpirun -np 8 python examples/02_all_load/all_load_bench.py -o /iris_results/all_load_bench.json
mpirun -np 8 python examples/03_all_store/all_store_bench.py -o /iris_results/all_store_bench.json


mpirun -np 8 python examples/04_atomic_add/atomic_add_bench.py  -o /iris_results/atomic_add_bench.json
mpirun -np 8 python examples/05_atomic_xchg/atomic_xchg_bench.py -o /iris_results/atomic_xchg_bench.json

mpirun -np 2 python examples/06_message_passing/message_passing_load_store.py 
mpirun -np 2 python examples/06_message_passing/message_passing_put.py

mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py --benchmark --validate -o /iris_results/gemm_all_scatter_bench.json
mpirun -np 8 python examples/08_gemm_atomics_all_reduce/benchmark.py --benchmark --validate -o /iris_results/gemm_atomics_all_reduce_bench.json
mpirun -np 8 python examples/09_gemm_one_shot_all_reduce/benchmark.py --benchmark --validate -o /iris_results/gemm_one_shot_all_reduce_bench.json
