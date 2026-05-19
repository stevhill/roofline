# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.test_utils import torch_dtype_map


def generate_golden_reference(input_length: int, R: int, dtype="bf16", seed=42):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]
    input_a = torch.rand(input_length, dtype=dtype_torch) * val_range

    output = input_a * 1.5 * R
    return {"A": input_a, "C": output}
