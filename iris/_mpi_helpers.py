from mpi4py import MPI
import numpy as np


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
