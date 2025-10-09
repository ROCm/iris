# Backend Interface

The backend provides two ways to work with GPU platforms:

For portable code, use the unified API that works across all backends:

```python
import iris.backend as backend

num_gpus = backend.count_devices()
backend.set_device(0)
ptr = backend.malloc(size)
```

For platform-specific code, import directly:

```python
from iris.backend import hip
ptr = hip.malloc_fine_grained(size)  # HIP-only: cache-coherent shared memory

from iris.backend import cuda
ptr = cuda.malloc_managed(size)  # CUDA-only: unified memory with page migration
```

## Implementing a New Backend

Backends must implement these functions:

```python
def set_device(gpu_id: int) -> None
def get_device_id() -> int
def count_devices() -> int
def get_cu_count(device_id: int | None = None) -> int
def get_wall_clock_rate(device_id: int) -> int
def get_arch_string(device_id: int | None = None) -> str
def get_num_xcc(device_id: int | None = None) -> int
def get_runtime_version() -> tuple[int, int]  # (major, minor)

def get_ipc_handle(ptr: int | ctypes.c_void_p, rank: int) -> Any
def open_ipc_handle(ipc_handle_data: np.ndarray, rank: int) -> int

def malloc(size: int) -> ctypes.c_void_p
def free(ptr: int | ctypes.c_void_p) -> None
```

See `cuda.py` and `hip.py` for reference implementations.
