# Debugging Common Issues

This guide helps you troubleshoot common problems when using Iris. Each section includes symptoms, causes, and solutions to get you back on track quickly.

## Installation Issues

### "ModuleNotFoundError: No module named 'iris'"

**Symptoms:**
```bash
python -c "import iris"
ModuleNotFoundError: No module named 'iris'
```

**Causes:**
- Iris not installed
- Wrong Python environment
- Installation failed

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep iris
   ```

2. **Reinstall Iris:**
   ```bash
   pip uninstall iris
   pip install -e .
   ```

3. **Check Python environment:**
   ```bash
   which python
   python --version
   ```

4. **Verify in container:**
   ```bash
   # If using Docker/Apptainer
   docker exec -it iris-dev bash
   python -c "import iris"
   ```

### "ImportError: cannot import name 'iris' from 'iris'"

**Symptoms:**
```bash
from iris import iris
ImportError: cannot import name 'iris' from 'iris'
```

**Causes:**
- Incomplete installation
- Version mismatch
- Corrupted installation

**Solutions:**

1. **Clean reinstall:**
   ```bash
   pip uninstall iris
   rm -rf build/ dist/ *.egg-info/
   pip install -e .
   ```

2. **Check source code:**
   ```bash
   ls iris/
   cat iris/__init__.py
   ```

3. **Verify dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Runtime Issues

### "CUDA out of memory"

**Symptoms:**
```bash
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Causes:**
- Heap size too large
- Buffer size too large
- Multiple processes sharing GPU memory

**Solutions:**

1. **Reduce heap size:**
   ```python
   # Reduce from 2GB to 1GB
   iris_ctx = iris.iris(heap_size=2**30)  # 1GB instead of 2GB
   ```

2. **Reduce buffer size:**
   ```python
   # Reduce buffer size
   buffer_size = 512  # Instead of 1024
   buffer = iris_ctx.zeros(buffer_size, dtype=torch.float32)
   ```

3. **Check GPU memory:**
   ```bash
   rocm-smi
   nvidia-smi  # If using CUDA
   ```

4. **Use smaller data types:**
   ```python
   # Use float16 instead of float32
   buffer = iris_ctx.zeros(1024, dtype=torch.float16)
   ```

### "MPI rank errors"

**Symptoms:**
```bash
mpirun -np 2 python script.py
# Only one process starts or errors about ranks
```

**Causes:**
- MPI not properly installed
- Wrong number of ranks
- GPU count mismatch

**Solutions:**

1. **Verify MPI installation:**
   ```bash
   which mpirun
   mpirun --version
   ```

2. **Check GPU count:**
   ```bash
   rocm-smi
   # Ensure you have enough GPUs for your rank count
   ```

3. **Match ranks to GPUs:**
   ```bash
   # For 2 GPUs, use 2 ranks
   mpirun -np 2 python script.py
   
   # For 4 GPUs, use 4 ranks
   mpirun -np 4 python script.py
   ```

4. **Test MPI separately:**
   ```bash
   mpirun -np 2 python -c "from mpi4py import MPI; print('MPI working')"
   ```

### "HIP errors" or "ROCm not found"

**Symptoms:**
```bash
RuntimeError: HIP error: invalid device ordinal
# or
RuntimeError: No HIP devices found
```

**Causes:**
- ROCm not installed
- GPU drivers not loaded
- Device not accessible

**Solutions:**

1. **Check ROCm installation:**
   ```bash
   rocm-smi
   hipconfig
   ```

2. **Verify GPU visibility:**
   ```bash
   export ROCR_VISIBLE_DEVICES=0,1,2,3
   rocm-smi
   ```

3. **Check device permissions:**
   ```bash
   ls -la /dev/dri/
   groups $USER
   ```

4. **Reinstall ROCm:**
   ```bash
   # Follow official ROCm installation guide
   # https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
   ```

## Communication Issues

### "Data not received correctly"

**Symptoms:**
```python
# Expected: tensor([42., 42., 42., 42.])
# Actual: tensor([0., 0., 0., 0.])
```

**Causes:**
- Missing barriers
- Wrong rank parameters
- Memory not allocated

**Solutions:**

1. **Add proper barriers:**
   ```python
   # Store data
   iris.store(buffer, data, 0, 1, heap_bases_ptr)
   iris_ctx.barrier()  # Wait for store to complete
   
   # Load data
   result = iris.load(buffer, 1, 0, heap_bases_ptr)
   ```

2. **Verify rank parameters:**
   ```python
   rank = iris_ctx.get_rank()
   print(f"Current rank: {rank}")
   
   # Ensure source_rank and target_rank are correct
   iris.store(buffer, data, source_rank=0, target_rank=1, heap_bases_ptr=heap_bases_ptr)
   ```

3. **Check memory allocation:**
   ```python
   # Ensure buffer is allocated on all ranks
   buffer = iris_ctx.zeros(1024, dtype=torch.float32)
   print(f"Buffer shape: {buffer.shape}, dtype: {buffer.dtype}")
   ```

### "Deadlock or hanging"

**Symptoms:**
```bash
# Program hangs, never completes
# Some ranks finish, others don't
```

**Causes:**
- Missing barriers
- Uneven execution paths
- MPI communication issues

**Solutions:**

1. **Add barriers at critical points:**
   ```python
   # After each major operation
   iris.store(buffer, data, 0, 1, heap_bases_ptr)
   iris_ctx.barrier()
   
   iris.load(buffer, 1, 0, heap_bases_ptr)
   iris_ctx.barrier()
   ```

2. **Ensure all ranks follow same path:**
   ```python
   # Good: All ranks execute same code
   iris_ctx.barrier()
   
   # Bad: Only some ranks execute
   if rank == 0:
       iris_ctx.barrier()  # Only rank 0 waits
   ```

3. **Check MPI timeout:**
   ```bash
   # Increase MPI timeout
   export OMPI_TIMEOUT=300
   mpirun -np 4 python script.py
   ```

## Performance Issues

### "Slow communication"

**Symptoms:**
```python
# Operations take much longer than expected
# Performance doesn't scale with GPU count
```

**Causes:**
- Small buffer sizes
- Too many barriers
- Inefficient patterns

**Solutions:**

1. **Increase buffer sizes:**
   ```python
   # Use larger buffers for better efficiency
   buffer_size = 8192  # Instead of 1024
   ```

2. **Reduce barriers:**
   ```python
   # Batch operations before barriers
   for i in range(10):
       iris.store(buffer + i*1024, data[i], 0, 1, heap_bases_ptr)
   
   iris_ctx.barrier()  # Single barrier for all operations
   ```

3. **Use efficient patterns:**
   ```python
   # Good: Vectorized operations
   offsets = tl.arange(0, 1024)
   iris.store(buffer + offsets, data, 0, 1, heap_bases_ptr)
   
   # Bad: Individual operations
   for i in range(1024):
       iris.store(buffer + i, data[i], 0, 1, heap_bases_ptr)
   ```

### "Memory usage too high"

**Symptoms:**
```python
# High GPU memory usage
# Out of memory errors
```

**Causes:**
- Large heap size
- Memory leaks
- Inefficient allocation

**Solutions:**

1. **Optimize heap size:**
   ```python
   # Start with smaller heap, increase as needed
   iris_ctx = iris.iris(heap_size=2**29)  # 512MB instead of 1GB
   ```

2. **Reuse buffers:**
   ```python
   # Reuse same buffer for multiple operations
   buffer = iris_ctx.zeros(1024, dtype=torch.float32)
   
   # Operation 1
   iris.store(buffer, data1, 0, 1, heap_bases_ptr)
   iris_ctx.barrier()
   
   # Operation 2 (reuse buffer)
   iris.store(buffer, data2, 0, 1, heap_bases_ptr)
   ```

3. **Use appropriate data types:**
   ```python
   # Use smallest dtype that meets precision requirements
   buffer = iris_ctx.zeros(1024, dtype=torch.float16)  # Instead of float32
   ```

## Debugging Tools

### Enable Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Use Iris logging
iris_ctx.debug("Starting operation")
iris_ctx.info("Operation completed")
iris_ctx.warning("High memory usage")
iris_ctx.error("Operation failed")
```

### Check System Status

```bash
# Check GPU status
rocm-smi

# Check MPI processes
ps aux | grep python

# Check memory usage
free -h
nvidia-smi  # If using CUDA
```

### Validate Results

```python
# Add validation checks
expected = torch.full((1024,), 42.0, dtype=torch.float32)
actual = iris.load(buffer, 1, 0, heap_bases_ptr)

if torch.allclose(expected, actual):
    print("✅ Data transfer successful")
else:
    print("❌ Data transfer failed")
    print(f"Expected: {expected[:5]}")
    print(f"Actual: {actual[:5]}")
```

## Common Patterns

### Debugging Template

```python
import iris
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

def debug_example():
    try:
        # Initialize Iris
        iris_ctx = iris.iris(heap_size=2**30)
        rank = iris_ctx.get_rank()
        
        iris_ctx.info(f"Rank {rank}: Iris initialized")
        
        # Allocate buffer
        buffer = iris_ctx.zeros(1024, dtype=torch.float32)
        iris_ctx.debug(f"Rank {rank}: Buffer allocated, shape: {buffer.shape}")
        
        # Your operations here
        if rank == 0:
            iris_ctx.info("Rank 0: Starting store operation")
            # Add your store operation
            
        iris_ctx.barrier()
        iris_ctx.info(f"Rank {rank}: Barrier completed")
        
        # Validation
        if rank == 1:
            iris_ctx.info("Rank 1: Starting load operation")
            # Add your load operation
            
        iris_ctx.barrier()
        iris_ctx.info(f"Rank {rank}: All operations completed")
        
    except Exception as e:
        iris_ctx.error(f"Rank {rank}: Error occurred: {e}")
        raise

if __name__ == "__main__":
    debug_example()
```

### Performance Profiling

```python
import time

def profile_operation():
    iris_ctx = iris.iris(heap_size=2**30)
    
    # Warm up
    buffer = iris_ctx.zeros(1024, dtype=torch.float32)
    iris_ctx.barrier()
    
    # Profile
    start_time = time.time()
    
    # Your operation here
    iris.store(buffer, 42.0, 0, 1, heap_bases_ptr)
    
    iris_ctx.barrier()
    end_time = time.time()
    
    print(f"Operation took {end_time - start_time:.4f} seconds")
```

## Getting Help

### Before Asking

1. **Check this guide**: Many issues are covered here
2. **Search issues**: Look for similar problems on GitHub
3. **Check discussions**: Search GitHub Discussions
4. **Verify environment**: Ensure your setup matches requirements

### When Asking for Help

Provide:

1. **Error message**: Complete error output
2. **Environment**: OS, Python version, GPU model, ROCm version
3. **Code**: Minimal example that reproduces the issue
4. **What you've tried**: Steps you've already attempted
5. **Expected vs actual**: What you expected vs what happened

### Helpful Commands

```bash
# System information
uname -a
python --version
pip list

# GPU information
rocm-smi
hipconfig

# MPI information
mpirun --version
which mpirun

# Iris information
python -c "import iris; print(iris.__version__)"
```

---

**Still having issues? Check the [Setup Alternatives](../getting-started/setup-alternatives.md) guide or start a discussion in GitHub Discussions!**
