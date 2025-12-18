# 22. RDMA Producer-Consumer

Producer-consumer pattern using InfiniBand RDMA for multi-node communication.

## Overview

This example demonstrates:
- Producer Triton kernel generates data on Rank 0
- RDMA transfer from Rank 0 to Rank 1
- Consumer Triton kernel verifies data on Rank 1

## Requirements

- InfiniBand network adapter
- libibverbs-dev installed
- Iris built with RDMA support

## Architecture

```
Rank 0 (Producer)              Rank 1 (Consumer)
─────────────────              ─────────────────
producer_kernel()              
     ↓ writes                  
GPU → CPU buffer               
     ↓                         
RDMA PUT ──────────────────→   CPU buffer
                                    ↓
                               CPU → GPU
                                    ↓
                               consumer_kernel()
                                    ↓ verifies
                               ✓ Success
```

## Usage

### Single Node (2 GPUs)
```bash
torchrun --nproc_per_node=2 rdma_producer_consumer.py
```

### Multi-Node (2 Nodes, 1 GPU each)
```bash
# Node 0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=<node0_ip> --master_port=29500 \
  rdma_producer_consumer.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=<node0_ip> --master_port=29500 \
  rdma_producer_consumer.py
```

## Expected Output

```
[Rank 0/2] Initialized on cuda:0
[Rank 1/2] Initialized on cuda:1
[Rank 0] Producing data
[Rank 0] First 10: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
[Rank 0] RDMA transfer to Rank 1
[Rank 0] RDMA completed
[Rank 1] Consuming data
[Rank 1] Received first 10: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
[Rank 1] Verified: 4096/4096
[Rank 1] SUCCESS!
```
