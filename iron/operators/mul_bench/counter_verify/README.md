# ROCm Counter Verification for gpu_ai_bench

This directory contains ROCm counter collection profiles and a helper script for the HIP GPU benchmark in IRON/iron/operators/mul_bench.

## Counter profile strategy

Two profiles are provided:

- fp16-fma profile (preferred)
  - file: rocprofv2_profile_fp16_fma.txt
  - counters:
    - SQ_INSTS_VALU_FMA_F16
    - GRBM_GUI_ACTIVE
    - SQ_WAVES
  - use when the direct FP16 FMA counter is exposed on the GPU.

- fallback-valu profile
  - file: rocprofv2_profile_fallback.txt
  - counters:
    - SQ_INSTS_VALU
    - GRBM_GUI_ACTIVE
    - SQ_WAVES
  - use when SQ_INSTS_VALU_FMA_F16 is not available.

The script auto-selects the profile by checking rocprofv2 --list-counters.

## FLOP mapping

For this benchmark kernel, v_pk_fma_f16 is the target instruction.

- expected FLOPs from benchmark formula:
  - flops = n * 2 * r * 2 * iters
- where:
  - n is half2 element count
  - 2 accounts for two fp16 lanes in half2
  - r is the FMA repeat count
  - 2 accounts for multiply plus add per FMA

If SQ_INSTS_VALU_FMA_F16 is available:

- FLOPs from direct counter can be computed as:
  - flops_from_counter = SQ_INSTS_VALU_FMA_F16 * 4
- because one packed FMA instruction updates two fp16 lanes and each lane performs 2 FLOPs.

If only fallback profile is available:

- SQ_INSTS_VALU is a proxy and includes non-FMA VALU instructions.
- use it for activity sanity-checks, not strict FLOP ground truth.

## Known caveats

- Counter availability is architecture and driver dependent.
- On some systems, direct FP16 FMA counters are not exposed through rocprofv2.
- AMD documentation notes Navi3x counter collection may require stable power state profile_standard for reliable values.
- Permission and kernel security settings can affect counter access.

## Files

- rocprofv2_profile_fp16_fma.txt
- rocprofv2_profile_fallback.txt
- rocprofv2_input.txt (backward-compatible fallback mirror)
- run_rocprofv2_verify.sh

## Usage

Run with auto profile selection:

```bash
cd IRON/iron/operators/mul_bench
./counter_verify/run_rocprofv2_verify.sh
```

Force direct profile:

```bash
./counter_verify/run_rocprofv2_verify.sh --profile fp16-fma
```

Force fallback profile:

```bash
./counter_verify/run_rocprofv2_verify.sh --profile fallback
```
