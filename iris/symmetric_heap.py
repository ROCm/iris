"""Compatibility shim — canonical code lives in iris.host.memory.symmetric_heap"""
from iris.host.memory.symmetric_heap import *  # noqa: F401,F403
from iris.host.memory.symmetric_heap import SymmetricHeap  # explicit for IDE
