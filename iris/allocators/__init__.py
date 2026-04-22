"""Compatibility shim — canonical code lives in iris.host.memory.allocators"""
from iris.host.memory.allocators import *  # noqa: F401,F403
from iris.host.memory.allocators.base import BaseAllocator  # noqa: F401
from iris.host.memory.allocators.torch_allocator import TorchAllocator  # noqa: F401
from iris.host.memory.allocators.vmem_allocator import VMemAllocator  # noqa: F401
