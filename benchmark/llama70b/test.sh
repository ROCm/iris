#!/bin/bash
# Reproduce the iris collective correctness bug: the gluon one-shot all_gather
# returns wrong results under cudagraph replay with varying inputs, while a
# torch.distributed control passes the same case. Runs against the baked exp
# stack (Dockerfile.exp). 8 GPUs, no model or server.
#
# Subset the matrix with COMMS_ARGS, e.g. COMMS_ARGS="-d bf16 -s 256,8192".
set -euo pipefail

cd /src/aiter
python3 op_tests/comms_tests/test_aiter_communicator.py ${COMMS_ARGS:-}
