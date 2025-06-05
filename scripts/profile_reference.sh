#!/bin/bash

# python_file=examples/gemm/reference/all_scatter.py
# python_file=examples/gemm/reference/all_gather.py
python_file=examples/gemm/reference/reduce_scatter.py
datatype=fp16
# datatype=fp32
num_gpus=8
M=8192
N=8192
K=30720
profile=false

export OMP_NUM_THREADS=1
rocprof_sys=/opt/rocprofiler-systems/bin/rocprof-sys-run

filename=$(basename "$python_file" .py)
output_base="results/reference_${filename}"
mkdir -p $output_base

# Final outputs
output_log_file="${output_base}.log"
rocprof_output="${output_base}"
output_json_file="${output_base}_${datatype}.json"
echo "Rocprof Output Prefix: ${rocprof_output}"

python_cmd="python -m torch.distributed.run --nproc_per_node=${num_gpus}"
python_cmd+=" ${python_file} -v -b -o ${output_json_file} -d ${datatype}"
python_cmd+=" -m ${M} -n ${N} -k ${K}"
echo "Python Command: ${python_cmd}"

if [ "$profile" = true ]; then
    echo "[INFO] Running with profiler enabled"

    ${rocprof_sys} \
    -T \
    --perfetto-annotations \
    --output "${rocprof_output}" \
    --use-rocm \
    --rocm-marker-api-operations roctxRangePushA roctxRangePop \
    -- ${python_cmd} 2>&1 | tee ${rocprof_output}/${output_log_file}

else
    echo "[INFO] Running without profiler"
    ${python_cmd}  2>&1 | tee ${output_log_file}
fi