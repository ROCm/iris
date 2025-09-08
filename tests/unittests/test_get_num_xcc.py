# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import iris

from test_utils import dist_spawn


@pytest.mark.parametrize(
    "num_calls",
    [
        10,
    ],
)
def test_get_num_xcc_api(request, num_calls):
    num_ranks = int(request.config.getoption("--num_ranks"))
    dist_spawn(_impl_test_get_num_xcc_api, num_ranks, num_calls)


def _impl_test_get_num_xcc_api(rank, world_size, num_calls):
    first = iris.hip.get_num_xcc()
    assert isinstance(first, int)
    for _ in range(num_calls):
        result = iris.hip.get_num_xcc()
        assert result == first, f"get_num_xcc changed between calls. Expected {first} but got {result}."
