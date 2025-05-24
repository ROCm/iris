# Put benchmark

Put benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/p2p/put/put_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional PUT bandwidth GB/s [Remote write]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   3657.58     49.39     48.98     49.20     48.56     49.01     49.29     49.03
GPU 01  ->     49.41   3616.00     49.39     48.81     48.50     48.60     49.35     48.94
GPU 02  ->     49.18     49.30   3607.16     49.09     48.06     47.87     48.75     48.23
GPU 03  ->     49.20     48.91     49.21   3552.08     47.92     48.15     48.62     48.58
GPU 04  ->     48.59     48.42     48.33     47.93   3656.06     49.21     49.31     49.06
GPU 05  ->     48.66     48.24     47.99     48.13     49.15   3534.84     49.43     48.96
GPU 06  ->     49.19     49.25     48.53     48.55     49.11     49.26   3571.96     49.20
GPU 07  ->     49.23     49.02     48.39     48.79     49.33     49.04     49.30   3670.31
```