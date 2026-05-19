#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from iron.operators.mul_bench.op import MulBench
from iron.operators.mul_bench.reference import generate_golden_reference
from iron.common.test_utils import verify_buffer


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "iron" / "operators" / "mul_bench" / "test.py").exists():
            return parent
    raise RuntimeError("Unable to locate repository root")


def build_artifact_paths(root: Path, size: int, r: int, num_columns: int, num_channels: int, tile_size: int) -> tuple[Path, Path]:
    stem = f"MulBench_sz{size}_c{num_columns}_ch{num_channels}_t{tile_size}_R{r}_npu2"
    return root / "build" / f"{stem}.xclbin", root / "build" / f"{stem}.bin"


def run_mul_bench(size: int, r: int, num_columns: int, num_channels: int, tile_size: int) -> int:
    root = find_repo_root()
    xclbin_path, insts_path = build_artifact_paths(root, size, r, num_columns, num_channels, tile_size)

    if not xclbin_path.exists() or not insts_path.exists():
        op = MulBench(
            size=size,
            tile_size=tile_size,
            num_aie_columns=num_columns,
            num_channels=num_channels,
            R=r,
        )
        op.compile()

    golden_ref = generate_golden_reference(input_length=size, R=r)
    input_buf = XRTTensor.from_torch(golden_ref["A"])
    output_buf = XRTTensor(golden_ref["C"].shape, dtype=bfloat16)

    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)

    # Create a start-marker file so external orchestrators know the NPU run started.
    started_marker = root / "build" / "mul_bench_npu.started"
    try:
        # Touch the marker before invoking the runtime so callers can detect start.
        started_marker.parent.mkdir(parents=True, exist_ok=True)
        started_marker.touch()
    except Exception:
        # Non-fatal: continue even if marker cannot be created
        pass

    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])

    # Remove the start marker after completion (best-effort)
    try:
        if started_marker.exists():
            started_marker.unlink()
    except Exception:
        pass

    errors = verify_buffer(output_buf.to_torch(), "output", golden_ref["C"])
    if errors:
        print(f"mul_bench verification failed with {len(errors)} errors")
        return 1

    print(f"NPU latency (us): {result.npu_time / 1e3:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mul_bench on the NPU using existing artifacts")
    parser.add_argument("--size", type=int, default=1 << 24)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--num-columns", type=int, default=8)
    parser.add_argument("--num-channels", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=8192)
    args = parser.parse_args()
    return run_mul_bench(args.size, args.r, args.num_columns, args.num_channels, args.tile_size)


if __name__ == "__main__":
    raise SystemExit(main())
