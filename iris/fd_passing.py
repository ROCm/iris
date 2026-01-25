# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Utilities for passing file descriptors between processes (Linux) using SCM_RIGHTS.

Torch distributed cannot transmit a live file descriptor by sending its integer
value because FD numbers are process-local. This module provides FD passing
for memory sharing between processes.
"""

from __future__ import annotations

import array
import os
import socket
import time
from typing import Dict, Tuple
from contextlib import contextmanager


@contextmanager
def managed_fd(fd: int):
    """
    Context manager for automatic FD cleanup.

    Args:
        fd: File descriptor to manage

    Yields:
        The file descriptor

    Example:
        >>> with managed_fd(my_fd) as fd:
        ...     send_fd(sock, fd)
        # FD is automatically closed here
    """
    try:
        yield fd
    finally:
        if fd >= 0:
            os.close(fd)


def send_fd(sock: socket.socket, fd: int, payload: bytes = b"\x00") -> None:
    """Send an FD over a connected Unix domain socket."""
    if fd < 0:
        raise ValueError(f"fd must be >= 0, got {fd}")
    fds = array.array("i", [fd])
    sock.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())])


def recv_fd(sock: socket.socket, payload_size: int = 1) -> Tuple[int, bytes]:
    """Receive an FD over a connected Unix domain socket."""
    fds = array.array("i")
    msg, ancdata, flags, addr = sock.recvmsg(payload_size, socket.CMSG_SPACE(fds.itemsize))
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: fds.itemsize])
            return fds[0], msg
    raise RuntimeError("No file descriptor received (missing SCM_RIGHTS)")


def make_rank_sock_path(prefix: str, rank: int) -> str:
    """Create a unique socket path for a rank."""
    # Keep paths short (AF_UNIX has a small path limit)
    return os.path.join("/tmp", f"{prefix}-{os.getpid()}-{rank}.sock")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before receiving all data")
        data += chunk
    return data


def setup_fd_mesh(rank: int, world_size: int, all_paths: Dict[int, str]) -> Dict[int, socket.socket]:
    """
    Create a simple persistent mesh:
    - Each rank listens on its own UDS path.
    - Each rank connects to all lower ranks (so exactly one connection per pair).
    - For a pair (i,j), the socket lives on the higher rank (j) and connects to i.

    Returns: dict peer_rank -> connected socket
    """
    # Listener for this rank
    path = all_paths[rank]
    try:
        os.unlink(path)
    except FileNotFoundError:
        # Socket path doesn't exist yet, no cleanup needed
        pass

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(world_size)

    conns: Dict[int, socket.socket] = {}

    # Connect to all lower ranks
    for peer in range(rank):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Lower-rank listener may not be bound yet; retry for a short period.
        deadline = time.time() + 10.0
        last_err: Exception | None = None
        while True:
            try:
                s.connect(all_paths[peer])
                break
            except FileNotFoundError as e:
                last_err = e
            except ConnectionRefusedError as e:
                last_err = e
            if time.time() >= deadline:
                raise FileNotFoundError(f"Timed out connecting rank {rank} -> {peer} at {all_paths[peer]}: {last_err}")
            time.sleep(0.01)
        # Identify ourselves to the server
        s.sendall(rank.to_bytes(4, "little", signed=False))
        conns[peer] = s

    # Accept connections from higher ranks
    for _ in range(rank + 1, world_size):
        client, _ = listener.accept()
        peer_rank_bytes = recv_exact(client, 4)
        peer_rank = int.from_bytes(peer_rank_bytes, "little", signed=False)
        conns[peer_rank] = client

    # Close listener and clean up socket path
    listener.close()
    try:
        os.unlink(path)
    except OSError:
        # Best effort cleanup
        pass

    return conns


def setup_fd_infrastructure(cur_rank: int, num_ranks: int):
    """
    Setup FD passing infrastructure for multi-rank communication.

    Creates Unix domain socket mesh for FD passing between ranks.

    Args:
        cur_rank: Current process rank
        num_ranks: Total number of ranks

    Returns:
        Dictionary mapping peer rank -> socket, or None for single rank
    """
    if num_ranks <= 1:
        return None

    import torch.distributed as dist
    from iris._distributed_helpers import distributed_barrier

    # Setup socket mesh for FD passing
    prefix = "iris-dmabuf"
    my_path = make_rank_sock_path(prefix, cur_rank)
    obj_list = [None for _ in range(num_ranks)]
    dist.all_gather_object(obj_list, my_path)
    all_paths = {r: obj_list[r] for r in range(num_ranks)}
    distributed_barrier()
    fd_conns = setup_fd_mesh(cur_rank, num_ranks, all_paths)
    distributed_barrier()

    return fd_conns
