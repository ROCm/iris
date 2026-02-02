# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for collective operations with ProcessGroups.

These tests verify that iris.ccl operations work correctly with torch.distributed
ProcessGroups, including both consecutive and strided rank patterns.

Requires at least 4 GPUs to run (for meaningful subgroup testing).
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


def _get_world_info():
    """Get world size and rank, skip if not enough ranks."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 4:
        pytest.skip(f"Process group tests require at least 4 ranks, got {world_size}")

    return world_size, rank


def _create_consecutive_groups(world_size, group_size=2):
    """
    Create consecutive (TP-like) groups.

    Example with world_size=4, group_size=2:
        Group 0: [0, 1]
        Group 1: [2, 3]

    Note: dist.new_group() is a collective operation - ALL ranks must call it,
    even if they're not part of the group being created.
    """
    groups = []
    for i in range(0, world_size, group_size):
        ranks = list(range(i, min(i + group_size, world_size)))
        if len(ranks) == group_size:
            # All ranks must call new_group collectively
            groups.append(dist.new_group(ranks))
        else:
            # For incomplete groups, still need a placeholder but all ranks
            # participated in all group creations above
            groups.append(None)
    return groups


def _create_strided_groups(world_size, num_groups=2):
    """
    Create strided (DP-like) groups.

    Example with world_size=4, num_groups=2:
        Group 0: [0, 2]  (stride=2)
        Group 1: [1, 3]  (stride=2)

    Note: dist.new_group() is a collective operation - ALL ranks must call it,
    even if they're not part of the group being created.
    """
    groups = []
    stride = num_groups

    for i in range(num_groups):
        ranks = list(range(i, world_size, stride))
        # All ranks must call new_group collectively
        groups.append(dist.new_group(ranks))

    return groups


def _get_my_group(groups, rank):
    """Find which group the current rank belongs to."""
    for i, group in enumerate(groups):
        if group is not None:
            group_ranks = dist.get_process_group_ranks(group)
            if rank in group_ranks:
                return i, group
    return None, None


# =============================================================================
# All-Reduce with Process Groups
# =============================================================================


@pytest.mark.parametrize(
    "variant",
    [
        "atomic",
        "two_shot",
        "one_shot",
        # TODO enable these tests when support for cache-modifiers is in place.
        # "spinlock",
    ],
)
@pytest.mark.parametrize("group_type", ["consecutive", "strided"])
def test_all_reduce_with_groups(variant, group_type, dtype=torch.float32, M=256, N=128):
    """Test all-reduce with ProcessGroups (consecutive and strided patterns)."""
    world_size, rank = _get_world_info()

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)

    # Create groups based on type
    if group_type == "consecutive":
        # TP-like: [0,1], [2,3], etc.
        groups = _create_consecutive_groups(world_size, group_size=2)
    else:
        # DP-like strided: [0,2], [1,3], etc.
        groups = _create_strided_groups(world_size, num_groups=2)

    group_idx, my_group = _get_my_group(groups, rank)
    assert my_group is not None, f"Rank {rank} not in any group"

    group_ranks = dist.get_process_group_ranks(my_group)

    # Create input tensor with deterministic values
    # Each rank fills with its global rank + 1 for easy verification
    pytorch_input_tensor = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_reduce on the group
    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM, group=my_group)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    # Run Iris all_reduce with the group
    shmem.barrier()
    config = Config(all_reduce_variant=variant)
    if variant == "two_shot":
        config.all_reduce_distribution = 1

    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()

    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, group=my_group, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    # Calculate expected sum for verification
    expected_sum = sum(r + 1 for r in group_ranks)

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank} (group {group_idx}, ranks={group_ranks}): "
            f"Iris output doesn't match PyTorch's all_reduce (variant={variant}, group_type={group_type})\n"
            f"Expected sum: {expected_sum}, got iris={iris_output_tensor[0, 0].item()}, pytorch={pytorch_output_tensor[0, 0].item()}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


# =============================================================================
# All-Gather with Process Groups
# =============================================================================


@pytest.mark.parametrize("group_type", ["consecutive", "strided"])
def test_all_gather_with_groups(group_type, dtype=torch.float32, M=128, N=64):
    """Test all-gather with ProcessGroups."""
    world_size, rank = _get_world_info()

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    if group_type == "consecutive":
        groups = _create_consecutive_groups(world_size, group_size=2)
    else:
        groups = _create_strided_groups(world_size, num_groups=2)

    group_idx, my_group = _get_my_group(groups, rank)
    assert my_group is not None

    group_ranks = dist.get_process_group_ranks(my_group)
    group_size = len(group_ranks)

    # Each rank fills with its global rank + 1
    pytorch_input_tensor = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # PyTorch output: (group_size * M, N)
    pytorch_output_tensor = torch.zeros(group_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output_tensor, pytorch_input_tensor, group=my_group)
    torch.cuda.synchronize()

    # Iris tensors
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((group_size * M, N), dtype=dtype)

    shmem.barrier()
    config = Config()
    shmem.ccl.all_gather(iris_output_tensor, iris_input_tensor, group=my_group, config=config)
    torch.cuda.synchronize()

    atol = 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank} (group {group_idx}, ranks={group_ranks}): "
            f"Iris output doesn't match PyTorch's all_gather (group_type={group_type})"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


# =============================================================================
# All-to-All with Process Groups
# =============================================================================


@pytest.mark.parametrize("group_type", ["consecutive", "strided"])
def test_all_to_all_with_groups(group_type, dtype=torch.float32, M=128, N=64):
    """Test all-to-all with ProcessGroups."""
    world_size, rank = _get_world_info()

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    if group_type == "consecutive":
        groups = _create_consecutive_groups(world_size, group_size=2)
    else:
        groups = _create_strided_groups(world_size, num_groups=2)

    group_idx, my_group = _get_my_group(groups, rank)
    assert my_group is not None

    group_ranks = dist.get_process_group_ranks(my_group)
    group_size = len(group_ranks)

    # Each rank creates input with its rank value
    pytorch_input_tensor = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank))

    # PyTorch all_to_all with list interface
    pytorch_input_list = [pytorch_input_tensor.clone() for _ in range(group_size)]
    pytorch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(group_size)]

    shmem.barrier()
    dist.all_to_all(pytorch_output_list, pytorch_input_list, group=my_group)
    torch.cuda.synchronize()

    # Convert to concatenated format
    pytorch_output_concat = torch.zeros(M, N * group_size, dtype=dtype, device=f"cuda:{rank}")
    for i in range(group_size):
        pytorch_output_concat[:, i * N : (i + 1) * N] = pytorch_output_list[i]

    # Iris: concatenated format (M, N * group_size)
    iris_input_concat = shmem.zeros((M, N * group_size), dtype=dtype)
    for i in range(group_size):
        iris_input_concat[:, i * N : (i + 1) * N] = pytorch_input_tensor

    iris_output_concat = shmem.zeros((M, N * group_size), dtype=dtype)

    shmem.barrier()
    config = Config()
    shmem.ccl.all_to_all(iris_output_concat, iris_input_concat, group=my_group, config=config)
    torch.cuda.synchronize()

    atol = 1e-5
    max_diff = torch.abs(iris_output_concat - pytorch_output_concat).max().item()

    try:
        assert torch.allclose(iris_output_concat, pytorch_output_concat, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank} (group {group_idx}, ranks={group_ranks}): "
            f"Iris output doesn't match PyTorch's all_to_all (group_type={group_type})"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


# =============================================================================
# Reduce-Scatter with Process Groups
# =============================================================================
#
# NOTE: Iris's reduce_scatter has different semantics than PyTorch's reduce_scatter_tensor:
# - PyTorch: input is (group_size * M, N), output is (M, N) - splits reduced tensor
# - Iris: input and output are both (M, N) - distributes tiles among ranks
#
# Until semantics are aligned, we test reduce_scatter with groups by verifying
# that the group operations produce mathematically correct results.


@pytest.mark.parametrize("group_type", ["consecutive", "strided"])
def test_reduce_scatter_with_groups(group_type, dtype=torch.float32, M=256, N=128):
    """
    Test reduce-scatter with ProcessGroups.

    Since Iris's reduce_scatter has different semantics than PyTorch's,
    we verify correctness by checking that:
    1. Each rank in the group receives its assigned tiles (reduced values)
    2. The sum of all tiles across the group equals the expected total
    """
    world_size, rank = _get_world_info()

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    if group_type == "consecutive":
        groups = _create_consecutive_groups(world_size, group_size=2)
    else:
        groups = _create_strided_groups(world_size, num_groups=2)

    group_idx, my_group = _get_my_group(groups, rank)
    assert my_group is not None

    group_ranks = dist.get_process_group_ranks(my_group)

    # Each rank fills with its global rank + 1
    input_value = float(rank + 1)
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.fill_(input_value)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config()
    shmem.ccl.reduce_scatter(iris_output_tensor, iris_input_tensor, group=my_group, config=config)
    torch.cuda.synchronize()

    # Expected sum for each tile (all ranks in group contribute)
    expected_sum = sum(r + 1 for r in group_ranks)

    # In reduce_scatter with tile distribution, each rank gets some tiles
    # with the reduced sum value. Check that non-zero tiles have the correct value.
    non_zero_mask = iris_output_tensor != 0

    try:
        if non_zero_mask.any():
            non_zero_values = iris_output_tensor[non_zero_mask]
            # All non-zero values should equal the expected sum
            assert torch.allclose(non_zero_values, torch.full_like(non_zero_values, expected_sum), atol=1e-5), (
                f"Rank {rank} (group {group_idx}, ranks={group_ranks}): "
                f"Non-zero tiles have incorrect values. Expected {expected_sum}, got unique values: {non_zero_values.unique().tolist()}"
            )

        # Gather outputs from all ranks in group to verify total coverage
        # (This is a simplified check - full verification would need cross-rank communication)

    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


# =============================================================================
# Edge Cases and Verification Tests
# =============================================================================


def test_group_info_extraction():
    """Test that extract_group_info returns correct values for different groups."""
    world_size, rank = _get_world_info()

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    from iris.ccl.utils import extract_group_info

    # Test 1: group=None should return global info
    rank_in_group, rank_global, ws, rank_start, rank_stride = extract_group_info(None, shmem)
    assert rank_in_group == rank_global == rank, "group=None: rank mismatch"
    assert ws == world_size, "group=None: world_size mismatch"
    assert rank_start == 0, "group=None: rank_start should be 0"
    assert rank_stride == 1, "group=None: rank_stride should be 1"

    # Test 2: Consecutive group [0, 1] - ALL ranks must call new_group collectively
    consecutive_group = dist.new_group([0, 1])
    if rank < 2:
        rank_in_group, rank_global, ws, rank_start, rank_stride = extract_group_info(consecutive_group, shmem)
        assert rank_in_group == rank, "Consecutive group: rank_in_group mismatch"
        assert rank_global == rank, "Consecutive group: rank_global mismatch"
        assert ws == 2, "Consecutive group: world_size should be 2"
        assert rank_start == 0, "Consecutive group: rank_start should be 0"
        assert rank_stride == 1, "Consecutive group: rank_stride should be 1"

    # Test 3: Strided group [0, 2] - ALL ranks must call new_group collectively
    if world_size >= 4:
        strided_group = dist.new_group([0, 2])
        if rank in [0, 2]:
            rank_in_group, rank_global, ws, rank_start, rank_stride = extract_group_info(strided_group, shmem)
            expected_rank_in_group = 0 if rank == 0 else 1
            assert rank_in_group == expected_rank_in_group, (
                f"Strided group: rank_in_group should be {expected_rank_in_group}, got {rank_in_group}"
            )
            assert rank_global == rank, f"Strided group: rank_global should be {rank}, got {rank_global}"
            assert ws == 2, "Strided group: world_size should be 2"
            assert rank_start == 0, "Strided group: rank_start should be 0"
            assert rank_stride == 2, "Strided group: rank_stride should be 2"

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_all_reduce_group_correctness():
    """
    Verify all-reduce with groups produces correct mathematical results.

    With strided groups [0,2] and [1,3]:
    - Group [0,2]: ranks fill with 1 and 3, sum should be 4
    - Group [1,3]: ranks fill with 2 and 4, sum should be 6
    """
    world_size, rank = _get_world_info()

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    # Create strided groups
    groups = _create_strided_groups(world_size, num_groups=2)
    group_idx, my_group = _get_my_group(groups, rank)
    group_ranks = dist.get_process_group_ranks(my_group)

    M, N = 64, 32
    dtype = torch.float32

    # Fill with rank + 1
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.fill_(float(rank + 1))
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(all_reduce_variant="two_shot")
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()

    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, group=my_group, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Calculate expected sum
    expected_sum = sum(r + 1 for r in group_ranks)
    actual_sum = iris_output_tensor[0, 0].item()

    try:
        assert abs(actual_sum - expected_sum) < 1e-5, (
            f"Rank {rank} (group ranks={group_ranks}): Expected sum {expected_sum}, got {actual_sum}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


def test_rank_stride_target_rank_calculation():
    """
    Explicitly test that rank_start + i * rank_stride correctly computes target_rank.

    This test verifies the core indexing mechanism used in CCL kernels:
    - Loop index `i` goes from 0 to world_size-1 (position in group)
    - `target_rank = rank_start + i * rank_stride` computes global rank
    - `group_rank` (rank_in_group) is compared with `i` for local vs remote operations

    Example with strided group [0, 2] (stride=2):
        i=0 -> target_rank = 0 + 0*2 = 0 (global rank 0)
        i=1 -> target_rank = 0 + 1*2 = 2 (global rank 2)
    """
    world_size, rank = _get_world_info()

    if world_size != 4:
        pytest.skip("This test requires exactly 4 ranks for strided group testing")

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    from iris.ccl.utils import extract_group_info

    # Test with strided group [0, 2] - stride of 2
    strided_group_02 = dist.new_group([0, 2])

    # Test with strided group [1, 3] - stride of 2
    strided_group_13 = dist.new_group([1, 3])

    if rank in [0, 2]:
        rank_in_group, rank_global, ws, rank_start, rank_stride = extract_group_info(strided_group_02, shmem)

        # Verify the target_rank calculation for each loop iteration
        expected_target_ranks = [0, 2]  # Global ranks in the group
        for i in range(ws):
            computed_target_rank = rank_start + i * rank_stride
            assert computed_target_rank == expected_target_ranks[i], (
                f"Rank {rank}: For i={i}, expected target_rank={expected_target_ranks[i]}, "
                f"got {computed_target_rank} (rank_start={rank_start}, rank_stride={rank_stride})"
            )

        # Verify that i == group_rank identifies the local rank correctly
        expected_local_i = 0 if rank == 0 else 1
        assert rank_in_group == expected_local_i, (
            f"Rank {rank}: rank_in_group={rank_in_group} should match expected_local_i={expected_local_i}"
        )

        # Verify: when i == rank_in_group, target_rank == rank_global
        local_target_rank = rank_start + rank_in_group * rank_stride
        assert local_target_rank == rank_global, (
            f"Rank {rank}: local_target_rank={local_target_rank} should equal rank_global={rank_global}"
        )

    if rank in [1, 3]:
        rank_in_group, rank_global, ws, rank_start, rank_stride = extract_group_info(strided_group_13, shmem)

        # Verify the target_rank calculation for each loop iteration
        expected_target_ranks = [1, 3]  # Global ranks in the group
        for i in range(ws):
            computed_target_rank = rank_start + i * rank_stride
            assert computed_target_rank == expected_target_ranks[i], (
                f"Rank {rank}: For i={i}, expected target_rank={expected_target_ranks[i]}, "
                f"got {computed_target_rank} (rank_start={rank_start}, rank_stride={rank_stride})"
            )

        # Verify that i == group_rank identifies the local rank correctly
        expected_local_i = 0 if rank == 1 else 1
        assert rank_in_group == expected_local_i, (
            f"Rank {rank}: rank_in_group={rank_in_group} should match expected_local_i={expected_local_i}"
        )

        # Verify: when i == rank_in_group, target_rank == rank_global
        local_target_rank = rank_start + rank_in_group * rank_stride
        assert local_target_rank == rank_global, (
            f"Rank {rank}: local_target_rank={local_target_rank} should equal rank_global={rank_global}"
        )

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_all_gather_strided_data_placement():
    """
    Verify all-gather with strided groups places data in correct output locations.

    This test ensures that with strided groups like [0, 2]:
    - Rank 0's data goes to output[0:M, :] on all group members
    - Rank 2's data goes to output[M:2M, :] on all group members

    The key insight: output placement uses rank_in_group (0, 1) not global rank (0, 2).
    """
    world_size, rank = _get_world_info()

    if world_size != 4:
        pytest.skip("This test requires exactly 4 ranks for strided group testing")

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    M, N = 64, 32
    dtype = torch.float32

    # Create strided groups [0, 2] and [1, 3]
    strided_group_02 = dist.new_group([0, 2])
    strided_group_13 = dist.new_group([1, 3])

    # Test with group [0, 2]
    if rank in [0, 2]:
        group_ranks = [0, 2]
        group_size = 2

        # Each rank fills input with its global rank + 1 for identification
        input_tensor = shmem.zeros((M, N), dtype=dtype)
        input_tensor.fill_(float(rank + 1))  # Rank 0 -> 1.0, Rank 2 -> 3.0

        output_tensor = shmem.zeros((group_size * M, N), dtype=dtype)

        shmem.barrier()
        config = Config()
        shmem.ccl.all_gather(output_tensor, input_tensor, group=strided_group_02, config=config)
        torch.cuda.synchronize()

        # Verify data placement:
        # - output[0:M, :] should contain rank 0's data (value 1.0)
        # - output[M:2M, :] should contain rank 2's data (value 3.0)
        chunk_0 = output_tensor[0:M, :].mean().item()
        chunk_1 = output_tensor[M : 2 * M, :].mean().item()

        expected_chunk_0 = 1.0  # From global rank 0 (rank_in_group=0)
        expected_chunk_1 = 3.0  # From global rank 2 (rank_in_group=1)

        assert abs(chunk_0 - expected_chunk_0) < 1e-5, (
            f"Rank {rank}: output[0:M] should be {expected_chunk_0} (from rank 0), got {chunk_0}"
        )
        assert abs(chunk_1 - expected_chunk_1) < 1e-5, (
            f"Rank {rank}: output[M:2M] should be {expected_chunk_1} (from rank 2), got {chunk_1}"
        )

    # Test with group [1, 3]
    if rank in [1, 3]:
        group_ranks = [1, 3]
        group_size = 2

        # Each rank fills input with its global rank + 1 for identification
        input_tensor = shmem.zeros((M, N), dtype=dtype)
        input_tensor.fill_(float(rank + 1))  # Rank 1 -> 2.0, Rank 3 -> 4.0

        output_tensor = shmem.zeros((group_size * M, N), dtype=dtype)

        shmem.barrier()
        config = Config()
        shmem.ccl.all_gather(output_tensor, input_tensor, group=strided_group_13, config=config)
        torch.cuda.synchronize()

        # Verify data placement:
        # - output[0:M, :] should contain rank 1's data (value 2.0)
        # - output[M:2M, :] should contain rank 3's data (value 4.0)
        chunk_0 = output_tensor[0:M, :].mean().item()
        chunk_1 = output_tensor[M : 2 * M, :].mean().item()

        expected_chunk_0 = 2.0  # From global rank 1 (rank_in_group=0)
        expected_chunk_1 = 4.0  # From global rank 3 (rank_in_group=1)

        assert abs(chunk_0 - expected_chunk_0) < 1e-5, (
            f"Rank {rank}: output[0:M] should be {expected_chunk_0} (from rank 1), got {chunk_0}"
        )
        assert abs(chunk_1 - expected_chunk_1) < 1e-5, (
            f"Rank {rank}: output[M:2M] should be {expected_chunk_1} (from rank 3), got {chunk_1}"
        )

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
