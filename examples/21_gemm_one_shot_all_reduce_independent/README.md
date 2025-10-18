# Example 21: GEMM One-Shot All-Reduce (Independent)

This example demonstrates independent execution of GEMM and all-reduce operations where:
1. GEMM and all-reduce work on completely separate data
2. No tile-based synchronization between operations
3. Both operations can overlap for maximum throughput

## Key Differences from Other Examples

Unlike examples 09 and 17 where GEMM and all-reduce are tightly coupled:
- **Independent Data**: GEMM operates on matrices A×B, all-reduce on separate tensors
- **No Synchronization**: Operations don't wait for each other
- **Maximum Overlap**: Both can execute concurrently on different SMs

## Usage

### Run both GEMM and all-reduce (default)
```bash
python benchmark.py -v --validate -b --benchmark
```

### Run only GEMM
```bash
python benchmark.py --only_gemm -v --validate
```

### Run only all-reduce
```bash
python benchmark.py --only_comm -v --validate
```

### CSV-based configuration sweep
```bash
python benchmark.py --csv configs.csv -b --benchmark
```

## CSV Format

Create a CSV file with configurations:

```csv
m,n,k,datatype,blk_m,blk_n,blk_k,gemm_sms,comm_sms
8192,4608,36864,fp16,256,64,64,256,48
8192,4096,12288,fp32,256,128,64,256,48
4096,4096,4096,bf16,128,128,64,240,56
```

## Key Features

- **Independent Operations**: GEMM and all-reduce work on separate data
- **Selective Execution**: `--only_gemm` or `--only_comm` flags
- **CSV Configuration**: Batch testing with multiple configurations
- **Template from Example 20**: Based on independent all-scatter example

## Parameters

### Matrix Dimensions
- `-m`: GEMM matrix A rows (default: 8192)
- `-n`: GEMM matrix B columns (default: 4608)
- `-k`: GEMM common dimension (default: 36864)
- `--m_comm`: All-reduce rows (default: same as -m)
- `--n_comm`: All-reduce columns (default: same as -n)

### Configuration
- `--BLK_M/N/K`: Block sizes for tiling
- `--gemm_sms`: Number of SMs for GEMM (default: 256)
- `--comm_sms`: Number of SMs for all-reduce (default: 48)
- `-r, --num_ranks`: Number of GPUs (default: 8)

### Execution Control
- `--only_gemm`: Run only GEMM operation
- `--only_comm`: Run only all-reduce operation
- `--csv`: Path to CSV configuration file

## Validation

The example includes validation for both operations:
- **GEMM**: Validates C = A × B
- **All-reduce**: Validates sum across all ranks (each rank contributes rank+1)

Expected all-reduce result: `sum(1...world_size) = world_size * (world_size + 1) / 2`
