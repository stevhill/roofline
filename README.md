
<!--
SPDX-FileCopyrightText: Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IRON — GPU + NPU Benchmarking Fork

> **Note:** This repository is a fork of [amd/iron](https://github.com/amd/iron), the open-source IRON NPU programming framework by Advanced Micro Devices.
> The upstream project provides a close-to-metal Python API for AMD Ryzen™ AI NPUs via MLIR-AIE bindings.
>
> This fork extends the upstream codebase with a combined **GPU + NPU benchmarking suite** (`mul_bench`), roofline analysis notebooks, and supporting tools for measuring and comparing GPU and NPU performance on the same workload.
> Upstream features, operators, and installation instructions are preserved as-is.

---

## Installation (Linux)

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 24.04** or **Ubuntu 24.10** install.

### Initial Setup

  > **Important**: Ensure your system has the latest BIOS version that enables NPU support. Check your laptop/mini-PC manufacturer's support website for BIOS updates.

If starting from `Ubuntu 24.04` you may need to update the Linux kernel to 6.11+ by installing the Hardware Enablement (HWE) stack:

  ```bash
  sudo apt update
  sudo apt install --install-recommends linux-generic-hwe-24.04
  sudo reboot
  ```

1. Install XDNA™ Driver and XRT:

    > [Instructions from mlir-aie repository](https://github.com/Xilinx/mlir-aie?tab=readme-ov-file#build-and-install-the-xdna-driver-and-xrt)

1. Install the packages needed for IRON and MLIR-AIE:

    ```bash
    # Python versions 3.10, 3.12 and 3.13 are currently supported by our wheels
    sudo apt install \
    build-essential clang clang-14 lld lld-14 python3-venv python3-pip
    ```

1. Setup a virtual environment and activate it:
   ```bash
   python3 -m venv ironenv
   source ironenv/bin/activate
   python3 -m pip install --upgrade pip
   ```

1. Source XRT (installed in step 1):
   ```bash
   source /opt/xilinx/xrt/setup.sh
   ```


### Building/Using & Testing Operators

All available operators can be found in `iron/operators`. These each contain:

- `op.py`: The Python operator interface -- an easy access point to integrate operators into your project that prescribes how to compile the operator (build artifacts) and how to call it at runtime (buffer sizes, etc.)
- `design.py`: The implementation of the operator's NPU code. Often references a kernel in `aie_kernels` for the compute core code and describes the data movement using ObjectFIFOs.
- `reference.py`: A reference CPU implementation to validate the correctness of the NPU implementation.
- `test.py`: An end-to-end test that instantiates and builds the operator, runs it and verifies its outputs against the reference.

> NOTE: Be sure the XRT setup script has been sourced and the Python environment is activated:
>       `source /opt/xilinx/xrt/setup.sh`
>       `source /path/to/ironenv/bin/activate`

To build and test all the operators:

``` bash
pytest iron/operators/ -m "not extensive"
```

To run the extensive test suite:

``` bash
pytest iron/operators/
```

To run a specific operator's tests:

``` bash
pytest iron/operators/axpy/
```

### Running mul_bench

The `mul_bench` benchmark can run GPU-only, NPU-only, or combined GPU+NPU execution.

#### Key file locations

| File | Purpose |
|:-----|:--------|
| [iron/operators/mul_bench/test.cpp](./iron/operators/mul_bench/test.cpp) | `main()` entry point — parses CLI flags, launches GPU+NPU threads in parallel |
| [iron/operators/mul_bench/gpu_ai_bench.cpp](./iron/operators/mul_bench/gpu_ai_bench.cpp) | GPU benchmark implementation (HIP kernel launch, timing) |
| [iron/operators/mul_bench/gpu_ai_bench.hpp](./iron/operators/mul_bench/gpu_ai_bench.hpp) | GPU benchmark interface |
| [iron/operators/mul_bench/npu_ai_bench.cpp](./iron/operators/mul_bench/npu_ai_bench.cpp) | NPU benchmark implementation (XRT runtime, timing) |
| [iron/operators/mul_bench/npu_ai_bench.hpp](./iron/operators/mul_bench/npu_ai_bench.hpp) | NPU benchmark interface |
| [iron/operators/mul_bench/run_mul_bench_npu.py](./iron/operators/mul_bench/run_mul_bench_npu.py) | Python NPU runner — builds artifacts and runs via IRON/XRT |
| [iron/operators/mul_bench/op.py](./iron/operators/mul_bench/op.py) | IRON operator interface (`MulBench` class) |
| [iron/operators/mul_bench/design.py](./iron/operators/mul_bench/design.py) | MLIR-AIE NPU kernel design |
| [iron/operators/mul_bench/reference.py](./iron/operators/mul_bench/reference.py) | CPU reference implementation for correctness checking |
| [iron/operators/mul_bench/test.py](./iron/operators/mul_bench/test.py) | pytest test suite |
| [iron/operators/mul_bench/Makefile](./iron/operators/mul_bench/Makefile) | Build system (compiles binary, manages NPU artifacts) |
| [iron/operators/mul_bench/mul_bench_parallel_test](./iron/operators/mul_bench/mul_bench_parallel_test) | Compiled benchmark binary (built by `make`) |
| [aie_kernels/generic/mul_bench.cc](./aie_kernels/generic/mul_bench.cc) | AIE compute kernel — vectorized bfloat16 element-wise multiply running on the AIE cores |



1. Make sure your environment is active:

   ```bash
   source /opt/xilinx/xrt/setup.sh
   source /path/to/ironenv/bin/activate
   ```

2. Build and run from the operator directory:

   ```bash
   cd iron/operators/mul_bench
   make run
   ```

   This will:
   - build `mul_bench_parallel_test`
   - generate required NPU artifacts in `build/` if missing
   - run the benchmark

3. Optional: override NPU artifact parameters at build/run time:

   ```bash
   cd iron/operators/mul_bench
   make run NPU_SIZE=1048576 NPU_R=2 NPU_COLUMNS=8 NPU_CHANNELS=2 NPU_TILE_SIZE=8192
   ```

4. Optional: run the binary directly with runtime flags:

   ```bash
   cd iron/operators/mul_bench
   ./mul_bench_parallel_test --help
   ./mul_bench_parallel_test --r 200 --iters-gpu 1 --iters-npu 1 --repeats 3 --json
   ```

5. Optional: run only the NPU Python path:

   ```bash
   python iron/operators/mul_bench/run_mul_bench_npu.py --size 1048576 --r 2 --num-columns 8 --num-channels 2 --tile-size 8192
   ```

#### Notebook sweeps and visualization

If you want to sweep `R` values and plot the results interactively, use the notebook in the repository root:

- [generate_roofline_gpu_npu_template.ipynb](./generate_roofline_gpu_npu_template.ipynb) for a lighter starting point you can customize

Typical notebook flow:

1. Open the notebook in VS Code or Jupyter.
2. Edit the configuration cell to set `R_VALUES`, `SIZE_GPU`, `SIZE_NPU`, `ITERS_GPU`, `ITERS_NPU`, and the NPU layout parameters.
3. Run the helper cells that build NPU artifacts and execute the benchmark sweeps.
4. Run the plotting cells to visualize GPU, NPU, and combined roofline points.
5. Save the generated plots or export the data once the sweep finishes.
