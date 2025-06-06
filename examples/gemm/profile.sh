#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


num_gpus=1
algorithm=one_shot_v2
m=8192
n=4608
k=36864
total_sms=304
streamk_sms=256
blk_m=256
blk_n=256
blk_k=64
gsize_m=6
COMMUNICATION_TILE_M=128
COMMUNICATION_TILE_N=64
num_stages=2
num_warps=8
communication_block_size=256
datatype="fp16"

timestamp=$(date +"%Y%m%d_%H%M%S")


# Build a descriptive base name
# output_base="results/${timestamp}/fp16_comm${COMMUNICATION_TILE_M}x${COMMUNICATION_TILE_N}x${communication_block_size}_${m}_${n}_${k}_streamksms${streamk_sms}_blk_m${blk_m}_blk_n${blk_n}_blk_k${blk_k}_gsize_m${gsize_m}"
output_base="results/${timestamp}/M${m}_N${n}_K${k}"

mkdir -p $output_base


# Final outputs
output_json_file="${output_base}.json"
output_cmd_file="${output_base}.cmd"
output_log_file="${output_base}.log"
proto_log_file="${output_base}.proto"
rocprof_output="${output_base}-%pid%"
rocprof_output="${output_base}"

# Echo for verification
echo "JSON Output: ${output_json_file}"
echo "Rocprof Output Prefix: ${rocprof_output}"

profile=false
benchmark=true
validate=true
debug=true
    # --benchmark --debug \

python_cmd="python benchmark.py \
    --algorithm ${algorithm} \
    -m ${m} -n ${n} -k ${k} \
    --total_sms ${total_sms} \
    --gemm_sms ${streamk_sms} \
    --BLK_M ${blk_m} --BLK_N ${blk_n} \
    --BLK_K ${blk_k} --gsize_m ${gsize_m} \
    --heap_size 8589934592 \
    --COMMUNICATION_TILE_M ${COMMUNICATION_TILE_M} \
    --COMMUNICATION_TILE_N ${COMMUNICATION_TILE_N} \
    --output_file ${output_json_file} \
    --num_stages ${num_stages} \
    --num_warps ${num_warps} \
    --datatype ${datatype}\
    --communication_block_size ${communication_block_size}\
    --benchmark"

if [ "$benchmark" = true ]; then
    python_cmd="${python_cmd} --benchmark"
fi
if [ "$debug" = true ]; then
    python_cmd="${python_cmd} --debug"
fi
if [ "$validate" = true ]; then
    python_cmd="${python_cmd} --validate"
fi
# export ROCPROFSYS_OUTPUT_FILE=$proto_log_file
# python_cmd="python vector_add.py"
# Launch
if [ "$profile" = true ]; then
    echo "[INFO] Running with profiler enabled"

    mpirun --allow-run-as-root -np ${num_gpus} /opt/rocprofiler-systems/bin/rocprof-sys-run \
    -T \
    --perfetto-annotations \
    --output "${rocprof_output}" \
    --use-rocm \
    --rocm-marker-api-operations roctxRangePushA roctxRangePop \
    -- ${python_cmd}

else
    echo "[INFO] Running without profiler"
    echo "mpirun --allow-run-as-root -np ${num_gpus} ${python_cmd}  2>&1 | tee ${output_log_file}" > ${output_cmd_file}
    mpirun --allow-run-as-root -np ${num_gpus} ${python_cmd}  2>&1 | tee ${output_log_file}
fi