# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

from mpi4py import MPI
import numpy as np
import torch


def mpi_allgather(data):
    thread_comm = MPI.COMM_WORLD
    shmcomm = thread_comm.Split_type(MPI.COMM_TYPE_SHARED)
    shm_size = shmcomm.Get_size()
    data = np.asarray(data)
    assert len(data.shape) == 1, "Only 1D arrays are supported."
    recv_data = np.empty(len(data) * shm_size, dtype=data.dtype)
    shmcomm.Allgather(sendbuf=data, recvbuf=recv_data)
    shmcomm.Free()
    reshaped = recv_data.reshape(shm_size, len(data))
    return reshaped


def mpi_allgather_multidim(data):
    thread_comm = MPI.COMM_WORLD
    shmcomm = thread_comm.Split_type(MPI.COMM_TYPE_SHARED)
    shm_size = shmcomm.Get_size()

    data_np = np.asarray(data)
    original_dtype_np = data_np.dtype

    if "float" in original_dtype_np.name or "bfloat" in original_dtype_np.name:
        data_for_mpi = data_np.astype(np.float32)
    else:
        data_for_mpi = data_np

    mpi_datatype = _get_mpi_datatype(data_for_mpi.dtype)

    recv_data = np.empty(data_for_mpi.size * shm_size, dtype=data_for_mpi.dtype)
    recv_data_mpi_datatype = _get_mpi_datatype(recv_data.dtype)

    shmcomm.Allgather(
        sendbuf=[data_for_mpi, data_for_mpi.size, mpi_datatype],
        recvbuf=[recv_data, recv_data.size // shm_size, recv_data_mpi_datatype],
    )
    shmcomm.Free()

    if "float" in original_dtype_np.name or "bfloat" in original_dtype_np.name:
        reshaped = recv_data.reshape(shm_size, data_np.size).astype(original_dtype_np)
    else:
        reshaped = recv_data.reshape(shm_size, data_np.size)

    return reshaped


def mpi_broadcast_scalar(value=None, root=0):
    thread_comm = MPI.COMM_WORLD
    shmcomm = thread_comm.Split_type(MPI.COMM_TYPE_SHARED)
    shm_rank = shmcomm.Get_rank()

    if shm_rank == root:
        assert value is not None, "Root must provide a value."
        value = np.array(value)
        dtype = value.dtype
    else:
        value = None
        dtype = None
    dtype = shmcomm.bcast(dtype, root=root)
    if shm_rank != root:
        value = np.empty(1, dtype=dtype)
    else:
        value = np.array([value], dtype=dtype)
    shmcomm.Bcast(value, root=root)
    shmcomm.Free()
    return value[0]


def world_barrier():
    MPI.COMM_WORLD.Barrier()


def init_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    return comm, rank, world_size


_DTYPE_TO_MPI_DTYPE = {
    np.dtype(np.float16): MPI.FLOAT,
    np.dtype(np.float32): MPI.FLOAT,
    np.dtype(np.float64): MPI.DOUBLE,
    np.dtype(np.int8): MPI.INT8_T,
    np.dtype(np.int16): MPI.INT16_T,
    np.dtype(np.int32): MPI.INT32_T,
    np.dtype(np.int64): MPI.INT64_T,
    np.dtype(np.uint8): MPI.UINT8_T,
    np.dtype(np.uint16): MPI.UINT16_T,
    np.dtype(np.uint32): MPI.UINT32_T,
    np.dtype(np.uint64): MPI.UINT64_T,
}


def _get_mpi_datatype(numpy_dtype):
    """Maps a numpy.dtype object to its corresponding mpi4py.MPI.Datatype."""

    if numpy_dtype.name == "bfloat16":
        return MPI.FLOAT

    mpi_dtype = _DTYPE_TO_MPI_DTYPE.get(numpy_dtype)
    if mpi_dtype is None:
        raise ValueError(
            f"Unsupported NumPy dtype for MPI: {numpy_dtype}. Name: {numpy_dtype.name}. Please add it to _DTYPE_TO_MPI_DTYPE map."
        )
    return mpi_dtype


def mpi_broadcast_tensor(value_to_broadcast=None, root=0):
    thread_comm = MPI.COMM_WORLD
    shmcomm = thread_comm.Split_type(MPI.COMM_TYPE_SHARED)
    shm_rank = shmcomm.Get_rank()

    original_dtype_name = None
    original_shape = None
    np_value_for_bcast = None

    if shm_rank == root:
        if value_to_broadcast is None:
            raise ValueError("Root must provide a value to broadcast.")

        if isinstance(value_to_broadcast, torch.Tensor):
            np_value_original = value_to_broadcast.cpu().numpy()
        else:
            np_value_original = np.asarray(value_to_broadcast)

        original_dtype_name = np_value_original.dtype.name
        original_shape = np_value_original.shape

        if "float" in original_dtype_name or "bfloat" in original_dtype_name:
            np_value_for_bcast = np_value_original.astype(np.float32)
        else:
            np_value_for_bcast = np_value_original

        original_dtype_name = shmcomm.bcast(original_dtype_name, root=root)
        original_shape = shmcomm.bcast(original_shape, root=root)
    else:
        original_dtype_name = shmcomm.bcast(None, root=root)
        original_shape = shmcomm.bcast(None, root=root)

        if "float" in original_dtype_name or "bfloat" in original_dtype_name:
            np_value_for_bcast = np.empty(original_shape, dtype=np.float32)
        else:
            np_value_for_bcast = np.empty(original_shape, dtype=np.dtype(original_dtype_name))

    mpi_datatype = _get_mpi_datatype(np_value_for_bcast.dtype)

    shmcomm.Bcast([np_value_for_bcast, np_value_for_bcast.size, mpi_datatype], root=root)
    shmcomm.Free()

    if "float" in original_dtype_name or "bfloat" in original_dtype_name:
        final_np_value = np_value_for_bcast.astype(np.dtype(original_dtype_name))
    else:
        final_np_value = np_value_for_bcast

    return final_np_value
