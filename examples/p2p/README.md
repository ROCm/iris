# Message Passing

A simple message passing example using Triton. This example demonstrates how to send and receive messages between two processes.

## Usage

```terminal
mpirun -np 2 python p2p/message_passing.py
```
Prints:
```terminal
[Iris] [1/2] Rank 1 is receiving data from rank 0.
[Iris] [0/2] Rank 0 is sending data to rank 1.
[Iris] [1/2] Rank 1 has finished sending/receiving data.
[Iris] [1/2] Validating output...
[Iris] [0/2] Rank 0 has finished sending/receiving data.
[Iris] [0/2] Validating output...
[Iris] [1/2] Validation successful.
```