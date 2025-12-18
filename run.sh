#!/bin/bash


export IRIS_RDMA_POLL_MAX_ATTEMPTS=1000 
export IRIS_LOG_LEVEL=DEBUG
export IRIS_DEBUG_DATA=1
torchrun --nproc_per_node=2 examples/24_rdma_atomic_add/rdma_atomic_add.py