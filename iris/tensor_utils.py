"""
Utility functions for tensor creation and manipulation.

Copyright (c) 2026 Advanced Micro Devices, Inc.
"""

import torch


class CUDAArrayInterface:
    """
    Wrapper for creating PyTorch tensors from raw GPU pointers using __cuda_array_interface__.

    This provides a clean interface for creating tensors from device pointers,
    which is useful for VMem allocations, imported DMA-BUF handles, and other
    scenarios where we need to wrap existing GPU memory.

    Args:
        ptr: GPU device pointer (integer)
        size_bytes: Size of the memory region in bytes
        dtype: PyTorch data type (default: torch.uint8)
        device: PyTorch device string (default: 'cuda')
        shape: Optional explicit shape tuple (default: inferred from size_bytes and dtype)

    Example:
        >>> ptr = 0x7f0000000000  # Some GPU pointer
        >>> size_bytes = 1024
        >>> wrapper = CUDAArrayInterface(ptr, size_bytes, dtype=torch.float32)
        >>> tensor = torch.as_tensor(wrapper, device='cuda')
    """

    def __init__(
        self,
        ptr: int,
        size_bytes: int,
        dtype: torch.dtype = torch.uint8,
        device: str = "cuda",
        shape: tuple = None,
    ):
        self.ptr = ptr
        self.size_bytes = size_bytes
        self.dtype = dtype
        self.device = device

        if shape is not None:
            self.shape = shape
        else:
            element_size = torch.tensor([], dtype=dtype).element_size()
            num_elements = size_bytes // element_size
            self.shape = (num_elements,)

        self.typestr = self._get_typestr(dtype)

    @staticmethod
    def _get_typestr(dtype: torch.dtype) -> str:
        """
        Convert PyTorch dtype to numpy-style typestr for __cuda_array_interface__.

        Format: <endianness><kind><size>
        - endianness: '<' (little), '>' (big), '|' (not applicable)
        - kind: 'f' (float), 'i' (signed int), 'u' (unsigned int), 'b' (bool)
        - size: bytes per element

        Reference: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        """
        typestr_map = {
            torch.float32: "<f4",
            torch.float64: "<f8",
            torch.float16: "<f2",
            torch.bfloat16: "<f2",
            torch.int8: "|i1",
            torch.int16: "<i2",
            torch.int32: "<i4",
            torch.int64: "<i8",
            torch.uint8: "|u1",
            torch.bool: "|b1",
        }

        if dtype not in typestr_map:
            raise ValueError(f"Unsupported dtype for CUDA array interface: {dtype}")

        return typestr_map[dtype]

    @property
    def __cuda_array_interface__(self) -> dict:
        """
        Provide __cuda_array_interface__ protocol for PyTorch interop.

        This allows PyTorch to create tensors directly from GPU pointers
        without copying data.

        Returns:
            dict: CUDA array interface dictionary with shape, typestr, data, and version
        """
        return {
            "shape": self.shape,
            "typestr": self.typestr,
            "data": (self.ptr, False),  # (pointer, read_only=False)
            "version": 3,
        }


def tensor_from_ptr(
    ptr: int,
    size_bytes: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    shape: tuple = None,
) -> torch.Tensor:
    """
    Create a PyTorch tensor from a raw GPU pointer.

    This is a convenience function that wraps CUDAArrayInterface and creates
    the tensor in one call.

    Args:
        ptr: GPU device pointer (integer)
        size_bytes: Size of the memory region in bytes
        dtype: PyTorch data type (default: torch.float32)
        device: PyTorch device string (default: 'cuda')
        shape: Optional explicit shape tuple (default: inferred from size_bytes and dtype)

    Returns:
        torch.Tensor: Tensor wrapping the GPU memory

    Example:
        >>> ptr = 0x7f0000000000
        >>> tensor = tensor_from_ptr(ptr, 4096, dtype=torch.float32)
        >>> print(tensor.shape)  # torch.Size([1024])
    """
    wrapper = CUDAArrayInterface(ptr, size_bytes, dtype, device, shape)
    return torch.as_tensor(wrapper, device=device)
