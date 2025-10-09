# Backend Interface

To implement a new backend, create a module that implements these functions:

## Required Functions

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

def malloc_fine_grained(size: int) -> ctypes.c_void_p
def malloc(size: int) -> ctypes.c_void_p
def free(ptr: int | ctypes.c_void_p) -> None
```

See `cuda.py` and `hip.py` for reference implementations.
