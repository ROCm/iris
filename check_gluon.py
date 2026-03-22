#!/usr/bin/env python3
"""Quick check of IrisDeviceCtx compatibility with gluon."""
from iris.experimental.iris_gluon import IrisDeviceCtx
print(f"IrisDeviceCtx type: {type(IrisDeviceCtx)}")
print(f"__init__ type: {type(IrisDeviceCtx.__init__)}")
print(f"initialize type: {type(IrisDeviceCtx.initialize)}")

# Check if __init__ is a JIT function
init_fn = IrisDeviceCtx.__init__
if hasattr(init_fn, 'fn'):
    print(f"__init__.fn: {init_fn.fn}")
    print("__init__ IS a JIT function - this will fail in gluon context")
else:
    print("__init__ is NOT a JIT function")

from iris.ccl.all_to_all import GLUON_AVAILABLE
print(f"all_to_all GLUON_AVAILABLE={GLUON_AVAILABLE}")
