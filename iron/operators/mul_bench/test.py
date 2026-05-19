#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
import aie.utils as aie_utils
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from iron.operators.mul_bench.op import MulBench
from iron.operators.mul_bench.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    max_aie_columns = aie_utils.get_current_device().cols
    input_lengths = [1 << 20]
    Rs = [200]

    params = []
    for input_length in input_lengths:
        for num_aie_columns in range(8, max_aie_columns + 1):
            tile_size = input_length // 32
            if tile_size * 32 != input_length:
                continue
            for R in Rs:
                # Determine if this is a regular test case
                is_regular = input_length == 2048 and R == 3
                marks = [] if is_regular else [pytest.mark.extensive]

                params.append(
                    pytest.param(
                        input_length,
                        num_aie_columns,
                        1,
                        R,
                        marks=marks,
                    )
                )
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_length,num_aie_columns,tile_size,R",
    get_params(),
)
def test_mul_bench(input_length, num_aie_columns, tile_size, R, aie_context):
    golden_ref = generate_golden_reference(input_length=input_length, R=R)

    operator = MulBench(
        size=input_length,
        tile_size=8192,
        num_aie_columns=num_aie_columns,
        num_channels=2,
        R=R,
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["A"],}
    output_buffers = {"output": golden_ref["C"]}
    print(f"\nRunning test with input_length={input_length}, num_aie_columns={num_aie_columns}, tile_size={tile_size}, R={R}")
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
    #assert errors, f"Test failed with errors"

if __name__ == "__main__":
    pytest.main([__file__])