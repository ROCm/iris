# Get benchmark

Get benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/p2p/get/get_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional GET bandwidth GiB/s [Remote read]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   3545.14     41.84     41.41     41.58     40.70     40.99     41.12     41.17
GPU 01  ->     41.65   3594.34     41.55     41.00     40.66     40.54     41.42     41.18
GPU 02  ->     41.67     41.68   3626.95     41.38     40.55     39.98     40.97     40.53
GPU 03  ->     41.95     40.95     41.46   3586.23     40.26     40.51     40.93     41.03
GPU 04  ->     40.74     40.48     40.21     40.22   3537.88     41.64     41.63     41.38
GPU 05  ->     40.90     40.77     40.25     40.38     41.54   3579.52     41.54     41.25
GPU 06  ->     41.29     41.72     40.91     41.08     41.73     41.70   3574.18     41.72
GPU 07  ->     41.47     41.45     40.77     40.77     41.56     41.16     41.71   3594.06
```