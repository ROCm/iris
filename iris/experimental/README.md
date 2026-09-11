# Experimental allocation providers

Adapters that let Iris device kernels run on memory allocated by another
runtime. Iris device code is unchanged: `iris.store`, `load` and `copy` take
`heap_bases` as a plain pointer argument and translate with

```
remote = peer_bases[to] + (ptr - peer_bases[local_rank])
```

so a provider's whole job is to hand Iris one `int64[num_ranks]` table per
allocation whose `local_rank` entry is that allocation's own base.

Neither provider is imported by `iris/experimental/__init__.py`, so `import
iris` never requires either dependency.

| Provider | Dependency | Scope |
| --- | --- | --- |
| `rocshmem_provider.py` | `rocshmem4py` | intra-node (IPC) |

## rocSHMEM provider

### Installing `rocshmem4py`

There is no prebuilt wheel — `rocshmem4py` is not on PyPI, not in the ROCm
nightly wheel indexes, and the `rocm-systems` release assets are source
tarballs. But pip builds it from source in a single command, given a rocSHMEM
install to build against:

```bash
CMAKE_PREFIX_PATH=<rocshmem-install> pip install \
  "rocshmem4py @ git+https://github.com/ROCm/rocm-systems.git#subdirectory=python/rocshmem"
```

Verified against a rocSHMEM 3.7.0 install: builds and installs
`rocshmem4py-0.1.0+rocshmem3.7.0-cp312-cp312-linux_x86_64.whl` with no other
environment set. `ROCSHMEM_HOME` is accepted as a convenience but is not
required; `CMAKE_PREFIX_PATH` is the documented mechanism and takes precedence
over both it and `ROCM_PATH`. It is forwarded to CMake as a cache variable
specifically so a rocSHMEM shipped under `/opt/rocm` cannot shadow the one you
asked for.

Two properties of the result are worth knowing:

- It **contains** rocSHMEM rather than depending on it at run time — rocSHMEM is
  statically linked into the extension module, and the version records which one
  (`0.1.0+rocshmem3.7.0`). Nothing needs to be on `LD_LIBRARY_PATH` afterwards.
- Consequently rocSHMEM's **build options are fixed when `rocshmem4py` is
  built**, not when it is used.

The wheel is CPython-ABI-tagged (`cp312` above), so build it with the interpreter
that will run it.

CI does the same pip install from source, but from a checkout it already has
rather than a `git+` URL — see `.github/scripts/install_rocshmem.sh`. Since it
has to build rocSHMEM itself anyway, taking both from one checkout keeps the core
and the bindings at the same revision; a `git+` URL would clone independently and
could drift, which `find_package` would not catch because it only compares
versions. If you are building both by hand, prefer the same: point
`pip install` at your `python/rocshmem` directory rather than at the URL.

### Building rocSHMEM first

The bindings do not build rocSHMEM: `CMakeLists.txt` does
`find_package(rocshmem 3.5.0 CONFIG REQUIRED)` with no `FetchContent`, so
**rocSHMEM 3.5.0 or newer must already be installed**.

**`USE_IPC` must be `ON`.** This provider gets its peer addresses from
`rocshmem_ptr`, which returns NULL unconditionally when IPC is compiled out. The
provider raises with that hint if every peer comes back NULL, rather than handing
Iris a table of zeros that would become wild pointers inside a kernel. Upstream
defaults it ON.

```bash
cmake -S "$ROCSHMEM_SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$ROCSHMEM_HOME" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DGPU_TARGETS=gfx950 \
  -DUSE_IPC=ON
cmake --build "$BUILD" --parallel
cmake --install "$BUILD"
```

If `GPU_TARGETS` is rejected as `invalid offload arch combinations: 'gfx950' and
'gfx950:sramecc+:xnack-'`, rocSHMEM's auto-detected arch and the one
`find_package(hip)` derives from the build host disagree; set
`ROCSHMEM_GPU_TARGETS='gfx950:sramecc+:xnack-'` to match.

### Verifying

```bash
python -c "import rocshmem4py; print(rocshmem4py.__file__)"
python tests/run_tests_distributed.py \
  tests/unittests/test_rocshmem_provider.py --num_ranks 2 -v
```

The tests skip rather than fail when `rocshmem4py` is absent, when fewer than 2
ranks are present, or when peers are not directly addressable, so they are inert
in an environment without rocSHMEM.

`tests/manual_rocshmem_provider.py` covers what the unit tests structurally
cannot: `run_tests_distributed.py` launches `torchrun` with `--nnodes=1`, so the
unit tests only ever see intra-node peers, where `rocshmem_ptr` resolves every
one. The manual script exercises the multi-node case, where `rocshmem_ptr`
returns NULL for remote peers and `SymmetricAddressMap.direct` is the thing under
test (`EXPECT_INDIRECT=1`).

### Running

rocSHMEM must be initialised before the provider is constructed; the caller owns
bootstrap and tensor lifetime:

```python
dist.init_process_group(backend="gloo")
rocshmem4py.init_rocshmem_by_uniqueid(dist.group.WORLD)
provider = RocshmemProvider()
```

Allocation and free are both **collective** — `rocshmem_free` is documented as
"a collective operation and must be called by all PEs" — so every rank must make
the same calls in the same order. That is why `free()` is explicit rather than
driven by garbage collection: `__del__` would run at whatever moment each rank
happened to collect, and ranks would hang instead of raising.
