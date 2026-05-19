# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar

from iron.common import (
    ChanneledUnaryOperator,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)


@dataclass
class MulBench(ChanneledUnaryOperator):
    """AIE-accelerated element-wise multiplication"""
    R: int = 4  # Number of times to repeat the multiplication in the kernel

    kernel_name: ClassVar[str] = "mul_bench"
    kernel_fn_name: ClassVar[str] = "eltwise_mul_bf16_vector"
    callback_fn: ClassVar[str] = "my_mul_bench"

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        # axpy.cc lives under aie_kernels/generic/ (not device-specific)
        return [
            KernelObjectArtifact(
                "mul_bench.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul_bench.cc"
                    )
                ],
            )
        ]


    def _mlir_callback_args(self):
        return super()._mlir_callback_args() + [self.R]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                self.callback_fn,
                tuple(self._mlir_callback_args()),
            ),
        )