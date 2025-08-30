# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import iris


@pytest.mark.parametrize(
    "num_calls",
    [
        10,
    ],
)
def test_get_num_xcd_api(num_calls):
    first = iris.hip.get_num_xcd()
    assert isinstance(first, int)
    for _ in range(num_calls):
        result = iris.hip.get_num_xcd()
        assert result == first, f"get_num_xcd changed between calls. Expected {first} but got {result}."
