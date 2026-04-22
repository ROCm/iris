"""Compatibility shim — canonical code lives in iris.host.memory.tensor_utils"""
from iris.host.memory.tensor_utils import *  # noqa: F401,F403
from iris.host.memory.tensor_utils import CUDAArrayInterface, tensor_from_ptr  # explicit
