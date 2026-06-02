#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ProfileConfig:
    name: str
    profile_file: Path
    counter_name: str
    flops_per_counter: float


def parse_r_values(r_values: str) -> List[int]:
    values: List[int] = []
    for token in r_values.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("No R values provided")
    return values


def build_r_values(args: argparse.Namespace) -> List[int]:
    if args.r_start is None and args.r_end is None and args.r_step is None:
        return parse_r_values(args.r_values)

    if args.r_start is None or args.r_end is None or args.r_step is None:
        raise ValueError("Range mode requires --r-start, --r-end, and --r-step")
    if args.r_step <= 0:
        raise ValueError("--r-step must be > 0")
    if args.r_end < args.r_start:
        raise ValueError("--r-end must be >= --r-start")

    values = list(range(args.r_start, args.r_end + 1, args.r_step))
    if not values:
        raise ValueError("Range generated no R values")
    return values


def run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def choose_profile(profile_arg: str, counter_dir: Path) -> ProfileConfig:
    profile_fp16 = counter_dir / "rocprofv2_profile_fp16_fma.txt"
    profile_fallback = counter_dir / "rocprofv2_profile_fallback.txt"

    fp16_cfg = ProfileConfig(
        name="fp16-fma",
        profile_file=profile_fp16,
        counter_name="SQ_INSTS_VALU_FMA_F16",
        flops_per_counter=4.0,
    )
    fallback_cfg = ProfileConfig(
        name="fallback-valu",
        profile_file=profile_fallback,
        counter_name="SQ_INSTS_VALU",
        flops_per_counter=4.0,
    )

    if profile_arg in {"fp16", "fp16-fma"}:
        return fp16_cfg
    if profile_arg in {"fallback", "valu"}:
        return fallback_cfg

    probe = run_cmd(["rocprofv2", "--list-counters"], cwd=counter_dir.parent)
    if "SQ_INSTS_VALU_FMA_F16" in probe.stdout:
        return fp16_cfg
    return fallback_cfg


def ensure_binary(mul_bench_dir: Path, bench_bin: Path) -> None:
    if bench_bin.exists():
        return
    build = run_cmd(["make", "mul_bench_parallel_test"], cwd=mul_bench_dir)
    if build.returncode != 0:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        raise RuntimeError("Failed to build mul_bench_parallel_test")


def parse_counter_csv(csv_path: Path, counter_name: str) -> tuple[int, float, float]:
    rows = 0
    counter_sum = 0.0
    wave_size: float | None = None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if counter_name not in row:
                raise KeyError(f"Counter column {counter_name} not found in {csv_path}")
            if "Wave_Size" not in row:
                raise KeyError(f"Wave_Size column not found in {csv_path}")

            row_wave_size = float(row["Wave_Size"])
            if wave_size is None:
                wave_size = row_wave_size
            elif row_wave_size != wave_size:
                raise RuntimeError(
                    f"Wave_Size changed within {csv_path}: {wave_size} -> {row_wave_size}"
                )

            rows += 1
            counter_sum += float(row[counter_name])

    if rows == 0:
        raise RuntimeError(f"No dispatch rows found in {csv_path}")
    if wave_size is None:
        raise RuntimeError(f"Could not determine Wave_Size from {csv_path}")

    return rows, counter_sum, wave_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep R values and compare expected vs counter-derived FLOPs")
    parser.add_argument("--r-values", default="1,2,4,8,16,32,64,128", help="Comma-separated R values (used when range args are not set)")
    parser.add_argument("--r-start", type=int, default=None, help="Range start for R sweep (inclusive)")
    parser.add_argument("--r-end", type=int, default=None, help="Range end for R sweep (inclusive)")
    parser.add_argument("--r-step", type=int, default=None, help="Range step for R sweep")
    parser.add_argument("--iters", type=int, default=1, help="Kernel launches per repeat")
    parser.add_argument("--size-gpu", type=int, default=1 << 24, help="GPU size in fp16 elements")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats")
    parser.add_argument("--profile", default="auto", choices=["auto", "fp16", "fp16-fma", "fallback", "valu"], help="Counter profile")
    parser.add_argument("--out-dir", default="counter_verify/out_sweep", help="Output directory for sweep data")
    args = parser.parse_args()

    counter_dir = Path(__file__).resolve().parent
    mul_bench_dir = counter_dir.parent
    bench_bin = mul_bench_dir / "mul_bench_parallel_test"
    out_dir = (mul_bench_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    r_values = build_r_values(args)
    profile = choose_profile(args.profile, counter_dir)

    ensure_binary(mul_bench_dir, bench_bin)

    n_half2 = args.size_gpu // 2
    summary_rows = []

    for r in r_values:
        run_dir = out_dir / f"R_{r}"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_name = f"flops_r{r}_{profile.name}"
        cmd = [
            "rocprofv2",
            "--plugin",
            "file",
            "--plugin-version",
            "2",
            "-i",
            str(profile.profile_file),
            "-d",
            str(run_dir),
            "-o",
            run_name,
            str(bench_bin),
            "--r",
            str(r),
            "--iters-gpu",
            str(args.iters),
            "--size-gpu",
            str(args.size_gpu),
            "--repeats",
            str(args.repeats),
            "--no-npu",
        ]

        result = run_cmd(cmd, cwd=mul_bench_dir)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise RuntimeError(f"rocprofv2 run failed for R={r}")

        csv_candidates = sorted(run_dir.glob("**/*.csv"))
        if not csv_candidates:
            raise RuntimeError(f"No CSV output found for R={r} under {run_dir}")
        csv_path = csv_candidates[0]

        dispatch_count, counter_sum, wave_size = parse_counter_csv(csv_path, profile.counter_name)
        expected_per_dispatch = float(n_half2) * 2.0 * float(r) * 2.0
        expected_total = expected_per_dispatch * float(dispatch_count)
        actual_total = counter_sum * profile.flops_per_counter * wave_size
        diff = actual_total - expected_total
        diff_pct = (diff / expected_total * 100.0) if expected_total != 0.0 else 0.0

        summary_rows.append(
            {
                "R": r,
                "profile": profile.name,
                "counter_name": profile.counter_name,
                "dispatch_count": dispatch_count,
                "wave_size": wave_size,
                "expected_flops_total": expected_total,
                "actual_flops_total": actual_total,
                "difference_flops": diff,
                "difference_percent": diff_pct,
                "csv_path": str(csv_path),
            }
        )

    summary_csv = out_dir / "flops_sweep_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "R",
            "profile",
            "counter_name",
            "dispatch_count",
            "wave_size",
            "expected_flops_total",
            "actual_flops_total",
            "difference_flops",
            "difference_percent",
            "csv_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"Summary written to {summary_csv}")
        print("matplotlib not available; skipping plot generation")
        return 0

    x = [row["R"] for row in summary_rows]
    expected_y = [row["expected_flops_total"] for row in summary_rows]
    actual_y = [row["actual_flops_total"] for row in summary_rows]
    diff_pct_y = [row["difference_percent"] for row in summary_rows]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0.plot(x, expected_y, marker="o", label="Expected FLOPs")
    ax0.plot(x, actual_y, marker="s", label=f"Actual FLOPs ({profile.counter_name} * wave_size)")
    ax0.set_ylabel("FLOPs per profiled run")
    ax0.set_title("Expected vs Actual FLOPs across R")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(x, diff_pct_y, marker="^", color="tab:red")
    ax1.axhline(0.0, color="black", linewidth=1)
    ax1.set_xlabel("R")
    ax1.set_ylabel("Difference (%)")
    ax1.set_title("Percent Difference: (Actual - Expected) / Expected")
    ax1.grid(True, alpha=0.3)

    plot_path = out_dir / "flops_sweep_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    print(f"Profile used: {profile.name}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
