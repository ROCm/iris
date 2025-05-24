# Get benchmark

Get benchmark using Iris.

## Usage

```terminal
mpirun -np 8 python examples/p2p/get/get_bench.py
```
On an MI300X, this example will run on 8 GPUs. It prints:
```terminal
Unidirectional GET bandwidth GB/s [Remote read]
 SRC\DST      GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
GPU 00  ->   4177.35     45.44     44.79     45.25     43.87     44.34     45.14     45.38
GPU 01  ->     45.25   4264.71     45.17     44.72     44.26     44.45     44.95     44.76
GPU 02  ->     45.17     45.40   4234.79     44.92     43.76     43.73     44.14     43.78
GPU 03  ->     45.29     44.50     44.88   4283.08     43.65     43.70     44.54     44.28
GPU 04  ->     44.42     43.78     43.80     43.84   4167.82     45.05     45.23     45.18
GPU 05  ->     44.52     44.28     43.65     43.83     44.92   4285.21     44.99     45.12
GPU 06  ->     44.71     45.15     44.31     44.49     45.35     44.96   4290.92     45.27
GPU 07  ->     44.95     44.61     44.01     44.20     45.17     45.13     45.39   4240.02
```