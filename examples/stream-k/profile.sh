# rocprofv2 --hip-api --roctx-trace -o omniwise --plugin perfetto python test_onnx.py
# rocprofv2 --hip-api --roctx-trace -o streamk_shmem --plugin perfetto mpirun -np 2 python shmem_distributed_gemm.py

/opt/rocprofiler-systems/bin/rocprof-sys-run --use-roctx -o streamk_shmem_trace  -- python all_reduce.py