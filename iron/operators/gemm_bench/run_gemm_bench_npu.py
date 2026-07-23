#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from iron.common.test_utils import run_test
from iron.operators.gemm.op import GEMM
from iron.operators.gemm.reference import generate_golden_reference


def make_operator(args: argparse.Namespace) -> GEMM:
    return GEMM(
        M=args.m,
        K=args.k,
        N=args.n,
        tile_m=args.tile_m,
        tile_k=args.tile_k,
        tile_n=args.tile_n,
        num_aie_columns=args.num_aie_columns,
        b_col_maj=args.b_col_maj,
        c_col_maj=args.c_col_maj,
        prio_accuracy=args.prio_accuracy,
        emulate_bf16_mmul_with_bfp16=args.emulate_bf16_mmul_with_bfp16,
    )


def build_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    golden_ref = generate_golden_reference(
        M=args.m,
        K=args.k,
        N=args.n,
        b_col_maj=args.b_col_maj,
        c_col_maj=args.c_col_maj,
    )
    input_buffers = {
        "A": golden_ref["input"].flatten(),
        "B": golden_ref["input_b"][0].flatten(),
    }
    output_buffers = {
        "C": golden_ref["output"][0].flatten(),
    }
    return input_buffers, output_buffers


def compute_metrics(args: argparse.Namespace, latency_us: float, bandwidth_gbps: float) -> dict:
    flops = float(2 * args.m * args.k * args.n * args.timed_iters)
    bytes_moved = float((args.m * args.k + args.k * args.n + args.m * args.n) * 2 * args.timed_iters)
    seconds = latency_us / 1e6
    average_ms = latency_us / 1000.0
    tflops = (flops / seconds / 1e12) if seconds > 0 else 0.0
    arith_intensity = (flops / bytes_moved) if bytes_moved > 0 else 0.0
    return {
        "average_ms": average_ms,
        "flops": flops,
        "bytes_moved": bytes_moved,
        "tflops": tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "arith_intensity": arith_intensity,
    }


def run_gemm_bench(args: argparse.Namespace) -> tuple[dict, dict[str, list[int]], GEMM]:
    operator = make_operator(args)
    input_buffers, output_buffers = build_inputs(args)
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )
    metrics = compute_metrics(args, latency_us, bandwidth_gbps)
    return metrics, errors, operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and run an NPU GEMM benchmark using the existing IRON GEMM operator"
    )
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--num-aie-columns", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=1)
    parser.add_argument("--rel-tol", type=float, default=0.005)
    parser.add_argument("--abs-tol", type=float, default=0.005)
    parser.add_argument("--b-col-maj", action="store_true")
    parser.add_argument("--c-col-maj", action="store_true")
    parser.add_argument("--prio-accuracy", action="store_true")
    parser.add_argument("--emulate-bf16-mmul-with-bfp16", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics, errors, operator = run_gemm_bench(args)

    result = {
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "tile_m": args.tile_m,
        "tile_k": args.tile_k,
        "tile_n": args.tile_n,
        "num_aie_columns": args.num_aie_columns,
        "b_col_maj": args.b_col_maj,
        "c_col_maj": args.c_col_maj,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "npu": metrics,
        "xclbin": str(operator.xclbin_artifact.filename),
        "insts": str(operator.insts_artifact.filename),
    }

    if errors:
        result["errors"] = {name: len(indexes) for name, indexes in errors.items()}

    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"NPU GEMM M={args.m} K={args.k} N={args.n} "
            f"tiles=({args.tile_m},{args.tile_k},{args.tile_n}) cols={args.num_aie_columns}"
        )
        print(f"  Time (ms):           {metrics['average_ms']:.3f}")
        print(f"  TFLOPS:              {metrics['tflops']:.6f}")
        print(f"  Bandwidth (GB/s):    {metrics['bandwidth_gbps']:.6f}")
        print(f"  Arith intensity:     {metrics['arith_intensity']:.6f} FLOPs/Byte")
        print(f"  XCLBIN:              {operator.xclbin_artifact.filename}")
        print(f"  INSTS:               {operator.insts_artifact.filename}")
        if errors:
            print(f"  Verification errors: {result['errors']}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
