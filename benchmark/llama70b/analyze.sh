#!/bin/bash
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

echo "=== preprocess -> data.csv ==="
python3 -u preprocess.py arms.json   # -u: stream progress live (a slow trace parse must not look hung)
echo "=== render report.ipynb ==="
if ! jupyter nbconvert --to notebook --execute --inplace report.ipynb; then
  echo "ERROR: report.ipynb render FAILED (data.csv is valid; open report.ipynb to debug)" >&2
  exit 1
fi
echo "done: data.csv + report.ipynb"
