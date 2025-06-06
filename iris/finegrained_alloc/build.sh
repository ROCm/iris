#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

name="finegrained_allocator"

basic_warnings="-Wall -Wextra -Werror"

strict_warnings="-pedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align \
-Woverloaded-virtual -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2"

std_flags="-std=c++17"
output_flags="-shared -fPIC -o lib${name}.so"

hipcc $basic_warnings $strict_warnings $std_flags $output_flags ${name}.hip
