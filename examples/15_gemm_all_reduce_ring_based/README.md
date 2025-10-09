# Ring-Based All-Reduce GEMM

This example implements a distributed GEMM using a ring-based all-reduce algorithm for multi-GPU communication.

## Algorithm Overview

The ring-based all-reduce algorithm distributes the GEMM computation across multiple GPUs by:

1. **Data Partitioning**: Split the K dimension of matrices A and B across GPUs
   - Each GPU i gets: `A[:, i*K/N:(i+1)*K/N]` and `B[i*K/N:(i+1)*K/N, :]`
   - Where N is the number of GPUs

2. **Local Computation**: Each GPU computes a partial result
   - `local_C_i = local_A_i @ local_B_i`

3. **Ring All-Reduce**: Accumulate partial results using a ring topology
   - GPUs form a logical ring: 0 → 1 → 2 → ... → N-1 → 0
   - In each step, each GPU:
     - Sends its accumulator to the next GPU
     - Receives data from the previous GPU
     - Adds received data to its accumulator
   - After N-1 steps, all GPUs have the complete result

## Synchronization

The ring all-reduce requires careful synchronization to ensure correctness:

### Memory Ordering Requirements

1. **Data Write Before Flag**: The data written to the ring buffer must be visible before the flag is set
2. **Flag Read Before Data Read**: The flag must be checked before reading data from the ring buffer

### Implementation

The implementation uses atomic operations with memory ordering semantics:

```python
# Producer: Send data to next rank
iris.store(ring_buffer + offset, acc, cur_rank, next_rank, heap_bases, mask=mask)
iris.atomic_add(locks + tile_id, 1, cur_rank, next_rank, heap_bases, 
                sem="release", scope="sys")

# Consumer: Wait for data from previous rank
result = 0
while result != 1:
    result = iris.atomic_cas(locks + tile_id, 1, 0, cur_rank, cur_rank, 
                             heap_bases, sem="acquire", scope="sys")
recv_tile = tl.load(ring_buffer + offset, mask=mask)
```

**Key Points:**
- `atomic_add` with `sem="release"` ensures data write completes before flag is visible
- `atomic_cas` with `sem="acquire"` ensures data is visible after flag is read
- `scope="sys"` ensures visibility across the entire system (all GPUs)
- `atomic_cas` atomically reads and resets the flag in one operation

## Running the Benchmark

```bash
# Validation with 2 GPUs
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2

# Benchmarking with 8 GPUs
python examples/15_gemm_all_reduce_ring_based/benchmark.py --benchmark --num_ranks 8

# Both validation and benchmarking
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --benchmark --num_ranks 2
```

## Parameters

- `-m`: Number of rows in matrix A (default: 8192)
- `-n`: Number of columns in matrix B (default: 4608)
- `-k`: Shared dimension K (default: 36864)
- `--validate`: Enable validation against expected result
- `--benchmark`: Enable performance benchmarking
- `--num_ranks`: Number of GPUs to use (default: 2)
- `--BLK_M/N/K`: Block sizes for tiling
- `--gemm_sms`: Number of SMs to use for GEMM
- `--datatype`: Data type (fp16, fp32, bf16)

## Known Issues

- The K dimension must be divisible by the number of ranks
- The N dimension must be divisible by the number of ranks
- Requires AMD GPUs with ROCm support
