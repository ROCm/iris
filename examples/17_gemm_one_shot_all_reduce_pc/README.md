# Example 17: GEMM One-Shot All-Reduce (Producer-Consumer)

This example demonstrates a producer-consumer pattern for one-shot all-reduce where:
1. Each GPU computes a subset of tiles (producer phase)
2. Waits for tiles to be ready
3. Loads tiles from all GPUs
4. Accumulates and scatters results to all GPUs

## Distribution Modes

The example supports two tile distribution strategies:

### Striding Distribution (`--distribution 0`)
- GPU i computes tiles: i, i+world_size, i+2*world_size, ...
- Better load balancing across irregular workloads
- Example: With 4 GPUs, GPU 0 gets tiles [0, 4, 8, 12, ...], GPU 1 gets [1, 5, 9, 13, ...]

### Block Distribution (`--distribution 1`)
- GPU i computes continuous block: [i*N/world_size, (i+1)*N/world_size)
- Better cache locality within each GPU
- Example: With 4 GPUs and 16 tiles, GPU 0 gets [0-3], GPU 1 gets [4-7], GPU 2 gets [8-11], GPU 3 gets [12-15]

## Usage

### Basic validation
```bash
python benchmark.py -v --validate
```

### Benchmark with striding distribution
```bash
python benchmark.py -b --benchmark --distribution 0
```

### Benchmark with block distribution
```bash
python benchmark.py -b --benchmark --distribution 1
```

### Custom configuration
```bash
python benchmark.py -v --validate -m 8192 -n 4608 -k 36864 --BLK_M 256 --BLK_N 256 --BLK_K 64
```

## Key Features

- **Producer-Consumer Pattern**: Each GPU produces a subset of tiles and consumes from all GPUs
- **Tile-based Synchronization**: Uses locks and remote atomic operations for coordination
- **Flexible Distribution**: Two distribution modes for different performance characteristics
- **Template from Example 15**: Based on ring-based all-reduce with tile distribution

## Parameters

- `-m`: Matrix A rows (default: 8192)
- `-n`: Matrix B columns (default: 4608)
- `-k`: Common dimension (default: 36864)
- `--distribution`: 0 for striding, 1 for block (default: 0)
- `--BLK_M/N/K`: Block sizes for tiling
- `--num_sms`: Number of SMs to use (default: 256)
- `-r, --num_ranks`: Number of GPUs (default: 2)
