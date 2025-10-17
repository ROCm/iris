# Example 20: Independent GEMM and All-Scatter

This example demonstrates independent GEMM (General Matrix Multiplication) and all-scatter operations with support for loading configurations from CSV files.

## Features

- **Independent Operations**: GEMM and all-scatter operations run independently on separate streams
- **CSV Configuration Support**: Load multiple configurations from a CSV file for batch benchmarking
- **Flexible Parameters**: Support for various matrix dimensions (m, n, k) and data types (fp16, fp32, bf16, int8)
- **Configurable Resources**: Adjustable SM allocation for GEMM and communication kernels

## Usage

### Single Configuration

Run a single benchmark with command-line arguments:

```bash
python benchmark.py --benchmark --validate --num_ranks 2 -m 8192 -n 4608 -k 36864 --datatype fp16
```

### CSV Configuration Sweep

Run multiple benchmarks using configurations from a CSV file:

```bash
python benchmark.py --benchmark --validate --num_ranks 2 --csv ../../dataset/gemm_config.csv
```

## CSV Format

The CSV file should contain the following columns:
- `m`: Number of rows in matrix A (GEMM)
- `n`: Number of columns in matrix B (GEMM)
- `k`: Common dimension between matrices A and B (GEMM)
- `datatype`: Data type for computation (fp16, fp32, bf16, or int8)

Example CSV file (`dataset/gemm_config.csv`):

```csv
m,n,k,datatype
8192,4608,36864,fp16
8192,4096,12288,fp32
8192,3584,14336,bf16
4096,4096,8192,fp16
2048,2048,4096,fp16
```

## Output

When using CSV configurations, each benchmark run generates a unique output file with the configuration parameters in the filename:

```
log_m8192_n4608_k36864_fp16.json
log_m8192_n4096_k12288_fp32.json
log_m8192_n3584_k14336_bf16.json
...
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-m` | int | 8192 | Number of rows in matrix A (GEMM) |
| `-n` | int | 4608 | Number of columns in matrix B (GEMM) |
| `-k` | int | 36864 | Common dimension between matrices A and B (GEMM) |
| `--m_comm` | int | m | Number of rows for communication tensor |
| `--n_comm` | int | n | Total number of columns for communication tensor |
| `--datatype` | str | fp16 | Datatype of computation (fp16, fp32, int8, bf16) |
| `--csv` | str | None | Path to CSV file with configurations |
| `--output_file` | str | log.json | Output file name |
| `--BLK_M` | int | 256 | Block size M |
| `--BLK_N` | int | 64 | Block size N |
| `--BLK_K` | int | 64 | Block size K |
| `--gsize_m` | int | 6 | L2-cache locality swizzle parameter |
| `--heap_size` | int | 1<<33 | Iris heap size |
| `--gemm_sms` | int | 256 | Number of SMs for GEMM algorithm |
| `--comm_sms` | int | 48 | Number of SMs for All-Scatter kernel |
| `-r, --num_ranks` | int | 2 | Number of ranks/processes |
| `-d, --debug` | flag | False | Enable debug mode |
| `-v, --validate` | flag | False | Enable validation mode |
| `-t, --trace_tiles` | flag | False | Enable tile-tracing mode |
| `-b, --benchmark` | flag | False | Enable benchmarking mode |

## Notes

- When using `--csv`, the values from the CSV file override the command-line arguments for `m`, `n`, `k`, and `datatype`
- All other parameters (block sizes, SM counts, etc.) are shared across all CSV configurations
- The `n` and `k` dimensions must be divisible by the number of ranks
