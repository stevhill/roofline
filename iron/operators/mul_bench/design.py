# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_mul_bench(
    dev,
    size,
    num_columns,
    num_channels,
    tile_size,
    trace_size,
    R,
    tile_cap=4096,
):
    if num_columns == 8 and num_channels == 2:
        return my_mul_bench_32(
            dev,
            size,
            num_columns,
            num_channels,
            tile_size,
            trace_size,
            R,
            tile_cap,
        )

    xfr_dtype = bfloat16
    line_size = tile_cap if tile_size > tile_cap else tile_size
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]

    # When tile_cap > 4096 (e.g. 8192), tiles may exceed a single 8 KB bank,
    # so the FIFO depth must shrink to 1 to avoid exceeding local memory.
    fifo_kwargs = {}
    if tile_cap > 4096:
        fifodepth = 1 if line_size > 4096 else 2
        fifo_kwargs = {"depth": fifodepth}

    # Calculate number of iterations per core
    total_cores = num_columns * num_channels
    if size % total_cores != 0:
        raise ValueError(
            f"Number of elements ({size}) must be divisible by total cores ({total_cores})."
        )
    per_core_elements = size // total_cores
    N_div_n = per_core_elements // line_size

    # Chunk size sent per DMA channel
    chunk = size // num_columns // num_channels

    # Dataflow with ObjectFifos
    of_ins = [
        ObjectFifo(line_type, name=f"in{i}_{j}", **fifo_kwargs)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(line_type, name=f"out{i}_{j}", **fifo_kwargs)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # External, binary kernel definition
    kernel_fcn = Kernel(
        "eltwise_mul_bf16_vector",
        "mul_bench.o",
        [line_type, line_type, np.int32, np.int32],
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, kernel_line):
        for _ in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel_line(elem_in, elem_out, R, line_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to perform the task
    # Large tile sizes (>4096) with LUT-based kernels need more stack space
    # than the default 1024 bytes due to spilled vector temporaries.
    worker_kwargs = {"stack_size": 0xD00} if line_size > 4096 else {}
    my_workers = [
        Worker(
            core_fn,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                kernel_fcn,
            ],
            **worker_kwargs,
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Create a TensorAccessPattern for each channel
    taps = [
        TensorAccessPattern(
            (1, size),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
        rt.start(*my_workers)

        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a_in,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b_out,
                    taps[i * num_channels + j],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    # Place components and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


def my_mul_bench_32(
    dev,
    size,
    num_columns,
    num_channels,
    tile_size,
    trace_size,
    R,
    tile_cap=4096,
):
    """32-core layout for an 8-column device using 2 external channels.

    Each external channel is split on a memory tile into 2 core-local streams,
    so the operator still uses `num_channels=2` while driving 32 compute cores
    in a 4x8 physical grid.
    """
    xfr_dtype = bfloat16
    if num_columns != 8 or num_channels != 2:
        raise ValueError(
            "my_mul_bench_32 expects num_columns=8 and num_channels=2."
        )

    n_aie_rows = 4
    total_external_streams = num_columns * num_channels
    total_cores = total_external_streams * 2
    if size % total_cores != 0:
        raise ValueError(
            f"Number of elements ({size}) must be divisible by 32 for the 32-core layout."
        )

    per_core_elements = size // total_cores
    per_channel_elements = per_core_elements * 2
    line_size = min(tile_cap, tile_size, per_core_elements)
    if per_core_elements % line_size != 0:
        raise ValueError(
            f"Per-core slice size ({per_core_elements}) must be divisible by tile size ({line_size})."
        )

    # Guard against MemTile overflow: 4 parent FIFOs (2 in + 2 out) per column
    # at depth=1 must fit within the 512 KB MemTile budget.
    max_parent_bytes = per_channel_elements * np.dtype(xfr_dtype).itemsize
    if 4 * max_parent_bytes > 512 * 1024:
        raise ValueError(
            f"per_channel_elements ({per_channel_elements}) too large for 512 KB MemTile "
            f"(4 × {max_parent_bytes // 1024} KB = {4 * max_parent_bytes // 1024} KB > 512 KB). "
            f"Reduce size or increase num_columns/num_channels."
        )

    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    channel_type = np.ndarray[(per_channel_elements,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]

    # Parent FIFOs must be depth=1 to stay within the 512 KB MemTile budget
    # (4 parents × per_channel_elements × 2 B approaches the limit for large sizes).
    # Child FIFOs can use depth=2 (ping-pong) when line_size ≤ 4096, adding only
    # 8 × line_size × 2 B ≈ 64 KB extra per column — well within budget.
    parent_fifo_depth = 1
    child_fifo_depth = 1 if line_size > 4096 else 2

    of_ins = [
        ObjectFifo(channel_type, name=f"in{i}_{j}", depth=parent_fifo_depth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(channel_type, name=f"out{i}_{j}", depth=parent_fifo_depth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    of_in_core = []
    of_out_core = []
    for idx in range(total_external_streams):
        col = idx // num_channels
        mem_tile = Tile(col=col, row=1)
        child_ins = of_ins[idx].cons().split(
            offsets=[0, per_core_elements],
            obj_types=[line_type, line_type],
            names=[f"in_{idx}_r0", f"in_{idx}_r1"],
            depths=[child_fifo_depth, child_fifo_depth],
            placement=mem_tile,
        )
        child_outs = of_outs[idx].prod().join(
            offsets=[0, per_core_elements],
            obj_types=[line_type, line_type],
            names=[f"out_{idx}_r0", f"out_{idx}_r1"],
            depths=[child_fifo_depth, child_fifo_depth],
            placement=mem_tile,
        )
        of_in_core.extend(child_ins)
        of_out_core.extend(child_outs)

    kernel_fcn = Kernel(
        "eltwise_mul_bf16_vector",
        "mul_bench.o",
        [line_type, line_type, np.int32, np.int32],
    )

    def core_fn(of_in, of_out, kernel_line):
        for _ in range_(per_core_elements // line_size):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel_line(elem_in, elem_out, R, line_size)
            of_in.release(1)
            of_out.release(1)

    worker_kwargs = {"stack_size": 0xD00} if line_size > 4096 else {}
    my_workers = []
    for col in range(num_columns):
        for channel in range(num_channels):
            base = (col * num_channels + channel) * 2
            for row_offset in range(n_aie_rows // 2):
                worker_idx = base + row_offset
                my_workers.append(
                    Worker(
                        core_fn,
                        [
                            of_in_core[worker_idx].cons(),
                            of_out_core[worker_idx].prod(),
                            kernel_fcn,
                        ],
                        **worker_kwargs,
                    )
                )

    taps = [
        TensorAccessPattern(
            (1, size),
            per_channel_elements * idx,
            [1, 1, 1, per_channel_elements],
            [0, 0, 0, 1],
        )
        for idx in range(total_external_streams)
    ]

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
        rt.start(*my_workers)

        tg = rt.task_group()
        for idx in range(total_external_streams):
            rt.fill(
                of_ins[idx].prod(),
                a_in,
                taps[idx],
                task_group=tg,
            )
        for idx in range(total_external_streams):
            rt.drain(
                of_outs[idx].cons(),
                b_out,
                taps[idx],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
