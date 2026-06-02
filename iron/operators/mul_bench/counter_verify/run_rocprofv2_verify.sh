#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MUL_BENCH_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR="$SCRIPT_DIR/out"
BENCH_BIN="$MUL_BENCH_DIR/mul_bench_parallel_test"

PROFILE_FP16="$SCRIPT_DIR/rocprofv2_profile_fp16_fma.txt"
PROFILE_FALLBACK="$SCRIPT_DIR/rocprofv2_profile_fallback.txt"
PROFILE_SELECTED=""
PROFILE_NAME=""

R=2
ITERS=1
SIZE_GPU=$((1 << 24))
REPEATS=3

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --r)
        R=$2; shift 2;
        ;;
      --iters)
        ITERS=$2; shift 2;
        ;;
      --size)
        SIZE_GPU=$2; shift 2;
        ;;
      --repeats)
        REPEATS=$2; shift 2;
        ;;
      --profile)
        case "$2" in
          fp16|fp16-fma)
            PROFILE_SELECTED="$PROFILE_FP16"
            PROFILE_NAME="fp16-fma"
            ;;
          fallback|valu)
            PROFILE_SELECTED="$PROFILE_FALLBACK"
            PROFILE_NAME="fallback-valu"
            ;;
          auto)
            ;;
          *)
            echo "Unknown profile: $2" >&2
            echo "Allowed values: auto, fp16-fma, fallback" >&2
            exit 1
            ;;
        esac
        shift 2;
        ;;
      --help|-h)
        echo "Usage: $0 [--r N] [--iters N] [--size N] [--repeats N] [--profile auto|fp16-fma|fallback]"
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 1
        ;;
    esac
  done
fi

cd "$MUL_BENCH_DIR"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Building benchmark binary..."
  make mul_bench_parallel_test
fi

if [[ -z "$PROFILE_SELECTED" ]]; then
  if rocprofv2 --list-counters 2>/dev/null | grep -q "SQ_INSTS_VALU_FMA_F16"; then
    PROFILE_SELECTED="$PROFILE_FP16"
    PROFILE_NAME="fp16-fma"
  else
    PROFILE_SELECTED="$PROFILE_FALLBACK"
    PROFILE_NAME="fallback-valu"
  fi
fi

n=$((SIZE_GPU / 2))
flops_expected=$(awk "BEGIN { print $n * 2 * $R * 2 * $ITERS }")

mkdir -p "$OUTPUT_DIR"

cat <<EOINFO
Running ROCm counter verification with:
  binary:       $BENCH_BIN
  profile:      $PROFILE_NAME
  counter file: $PROFILE_SELECTED
  output dir:   $OUTPUT_DIR
  size_gpu:     $SIZE_GPU
  r:            $R
  iters:        $ITERS
  repeats:      $REPEATS
  expected FLOPs per launch set: $flops_expected
EOINFO

rocprofv2 -i "$PROFILE_SELECTED" -d "$OUTPUT_DIR" -o "flops_verify_profile_${PROFILE_NAME}" \
  "$BENCH_BIN" --r "$R" --iters-gpu "$ITERS" --size-gpu "$SIZE_GPU" --repeats "$REPEATS" --no-npu

echo
cat <<EOINFO
Collected ROCm counters in: $OUTPUT_DIR
Expected FLOPs formula: n * 2 * r * 2 * iters
  n = $n half2 values, r = $R, iters = $ITERS
  expected flops = $flops_expected

Profile used: $PROFILE_NAME
- fp16-fma: direct instruction counter SQ_INSTS_VALU_FMA_F16 (preferred)
- fallback-valu: SQ_INSTS_VALU proxy when direct FP16 FMA counter is unavailable

Inspect the ROCm output files for counter values.
EOINFO
