import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def atomic_or_kernel(
    results,
    sem: tl.constexpr,
    scope: tl.constexpr,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    val = 1 << (cur_rank % results.type.element_ty.primitive_bitwidth)
    acc = tl.full([BLOCK_SIZE], val, dtype=results.type.element_ty)

    for target_rank in range(num_ranks):
        iris.atomic_or(results + offsets, acc, cur_rank, target_rank, heap_bases, mask, sem=sem, scope=scope)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize(
    "sem",
    [
        "acquire",
        "release",
        "acq_rel",
    ],
)
@pytest.mark.parametrize(
    "scope",
    [
        "cta",
        "gpu",
        "sys",
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
def test_atomic_or_rank_aware(dtype, sem, scope, BLOCK_SIZE):
    # TODO: Adjust heap size.
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    cur_rank = shmem.get_rank()

    results = shmem.zeros(BLOCK_SIZE, dtype=dtype)

    grid = lambda meta: (1,)
    atomic_or_kernel[grid](results, sem, scope, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    shmem.barrier()

    bit_width = 32 if dtype == torch.int32 else 64
    effective_bits = min(num_ranks, bit_width)
    expected_scalar = (1 << effective_bits) - 1
    expected = torch.full((BLOCK_SIZE,), expected_scalar, dtype=dtype, device="cuda")

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(e)
        print("Expected:", expected)
        print("Actual  :", results)
        raise
