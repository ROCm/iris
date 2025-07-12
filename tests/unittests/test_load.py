import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def load_kernel(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)

    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE

    result = tl.zeros([BLOCK_SIZE], dtype=data.type.element_ty)
    for target_rank in range(num_ranks):
        result += iris.load(data + offsets, source_rank, target_rank, heap_bases, mask=mask)

    # Store data to result buffer
    tl.store(results + offsets, result, mask=mask)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "BLOCK_SIZE",
    [
        1,
        8,
        16,
        32,
    ],
)
def test_load_api(dtype, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()

    data = shmem.ones(BLOCK_SIZE, dtype=dtype)
    results = shmem.zeros_like(data)

    grid = lambda meta: (1,)
    load_kernel[grid](data, results, source_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=dtype, device="cuda") * num_ranks

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
