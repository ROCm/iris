# 23. RDMA Consumer Pull (GET)

Consumer-pull pattern using InfiniBand RDMA GET operations for multi-node communication.

## Overview

This example demonstrates:
- Rank 1 (Server) prepares data in its heap
- Rank 0 (Client) uses RDMA GET to pull data from Rank 1
- Triton kernel verifies pulled data on Rank 0

**Key Difference from Example 22:**
- **Example 22 (PUT)**: Sender initiates - Rank 0 pushes data to Rank 1
- **Example 23 (GET)**: Receiver initiates - Rank 0 pulls data from Rank 1

## Requirements

- InfiniBand network adapter
- libibverbs-dev installed
- Iris built with RDMA support

## Architecture

```
Rank 1 (Server)                Rank 0 (Client)
───────────────                ───────────────
Data in heap                   
     ↓                         
CPU buffer                     
                               RDMA GET ←──────────┐
                                                   │
CPU buffer ←───────────────────────────────────────┘
     ↓
CPU → GPU
     ↓
verify_kernel()
     ↓ verifies
✓ Success
```

## Usage

### Single Node (2 GPUs)
```bash
torchrun --nproc_per_node=2 rdma_consumer_pull.py
```

### Multi-Node (2 Nodes, 1 GPU each)
```bash
# Node 0 (Client - pulls data)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=<node0_ip> --master_port=29500 \
  rdma_consumer_pull.py

# Node 1 (Server - provides data)
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=<node0_ip> --master_port=29500 \
  rdma_consumer_pull.py
```

## Expected Output

```
[Rank 0/2] Initialized on cuda:0
[Rank 1/2] Initialized on cuda:1
[Rank 1] Server: Preparing Data
[Rank 1] Data ready, first 10: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
[Rank 0] Client: Pulling Data via RDMA GET
[Rank 0] RDMA GET operations enqueued to queue
[Rank 0] Barrier complete, all RDMA operations finished
[Rank 0] Received data first 10: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
[Rank 0] Verified: 4091/4091
============================================================
[Rank 0] SUCCESS! Data pulled correctly via RDMA GET!
```

## RDMA GET vs PUT

### When to use GET:
- **Consumer-initiated**: Receiver decides when to pull data
- **Pull-based flow control**: Consumer controls rate
- **Useful for**: Demand-driven workloads, load balancing

### When to use PUT:
- **Producer-initiated**: Sender decides when to push data
- **Push-based flow control**: Producer controls rate
- **Useful for**: Pipeline parallelism, streaming workloads

