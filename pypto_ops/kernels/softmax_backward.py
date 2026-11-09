# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Softmax backward — dx = y * (dy - row_sum(dy * y)); bf16 and fp32 paths, shape-dispatched."""
from __future__ import annotations


import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


ROWS = 512
COLS = 1024
REQUESTED_ROW_TILE = 32
ROW_TILE = REQUESTED_ROW_TILE
COL_TILE = 512
FULL_CHUNKS = COLS // COL_TILE
TAIL_COLS = COLS % COL_TILE
LONG_ROW_TILE = 16
PIPELINE_STAGE = 2
LONG_MIN_FULL_CHUNKS = 12
LONG_REDUCE_TILE = 1024
LONG_REDUCE_CHUNKS = COLS // LONG_REDUCE_TILE
LONG_REDUCE_TAIL = COLS % LONG_REDUCE_TILE


@pl.jit
def bf16_streamed_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.BF16],
    y: pl.Tensor[[ROWS, COLS], pl.BF16],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.BF16]],
):
    for row in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bf16_streamed_rows"):
            dot = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for chunk in pl.range(FULL_CHUNKS):
                col = chunk * COL_TILE
                grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                partial = pl.reshape(pl.row_sum(pl.mul(grad, saved)), [1, ROW_TILE])
                dot = pl.add(dot, partial)
            if TAIL_COLS > 0:
                start = FULL_CHUNKS * COL_TILE
                grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
                saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
                partial_tail = pl.reshape(pl.row_sum(pl.mul(grad_tail, saved_tail)), [1, ROW_TILE])
                dot = pl.add(dot, partial_tail)
            scale = pl.reshape(dot, [ROW_TILE, 1])
            for chunk in pl.range(FULL_CHUNKS):
                col = chunk * COL_TILE
                grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                result = pl.mul(saved, pl.row_expand_sub(grad, scale))
                dx[row : row + ROW_TILE, col : col + COL_TILE] = pl.cast(result, pl.BF16)
            if TAIL_COLS > 0:
                start = FULL_CHUNKS * COL_TILE
                grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
                saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
                result_tail = pl.mul(saved_tail, pl.row_expand_sub(grad_tail, scale))
                dx[row : row + ROW_TILE, start:COLS] = pl.cast(result_tail, pl.BF16)
    return dx


@pl.jit
def bf16_spmd_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.BF16],
    y: pl.Tensor[[ROWS, COLS], pl.BF16],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.BF16]],
):
    for block in pl.spmd(ROWS // ROW_TILE, name_hint="bf16_spmd_rows"):
        row = block * ROW_TILE
        dot = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(FULL_CHUNKS):
            col = chunk * COL_TILE
            grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
            saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
            partial = pl.reshape(pl.row_sum(pl.mul(grad, saved)), [1, ROW_TILE])
            dot = pl.add(dot, partial)
        if TAIL_COLS > 0:
            start = FULL_CHUNKS * COL_TILE
            grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
            saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
            partial_tail = pl.reshape(pl.row_sum(pl.mul(grad_tail, saved_tail)), [1, ROW_TILE])
            dot = pl.add(dot, partial_tail)
        scale = pl.reshape(dot, [ROW_TILE, 1])
        for chunk in pl.range(FULL_CHUNKS):
            col = chunk * COL_TILE
            grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
            saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
            result = pl.mul(saved, pl.row_expand_sub(grad, scale))
            dx[row : row + ROW_TILE, col : col + COL_TILE] = pl.cast(result, pl.BF16)
        if TAIL_COLS > 0:
            start = FULL_CHUNKS * COL_TILE
            grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
            saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
            result_tail = pl.mul(saved_tail, pl.row_expand_sub(grad_tail, scale))
            dx[row : row + ROW_TILE, start:COLS] = pl.cast(result_tail, pl.BF16)
    return dx


@pl.jit
def bf16_pipelined_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.BF16],
    y: pl.Tensor[[ROWS, COLS], pl.BF16],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.BF16]],
):
    for row in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bf16_pipelined_rows"):
            dot = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for chunk in pl.range(FULL_CHUNKS):
                col = chunk * COL_TILE
                grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                partial = pl.reshape(pl.row_sum(pl.mul(grad, saved)), [1, ROW_TILE])
                dot = pl.add(dot, partial)
            if TAIL_COLS > 0:
                start = FULL_CHUNKS * COL_TILE
                grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
                saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
                partial_tail = pl.reshape(pl.row_sum(pl.mul(grad_tail, saved_tail)), [1, ROW_TILE])
                dot = pl.add(dot, partial_tail)
            scale = pl.reshape(dot, [ROW_TILE, 1])
            for chunk in pl.pipeline(FULL_CHUNKS, stage=PIPELINE_STAGE):
                col = chunk * COL_TILE
                grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
                result = pl.mul(saved, pl.row_expand_sub(grad, scale))
                dx[row : row + ROW_TILE, col : col + COL_TILE] = pl.cast(result, pl.BF16)
            if TAIL_COLS > 0:
                start = FULL_CHUNKS * COL_TILE
                grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
                saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
                result_tail = pl.mul(saved_tail, pl.row_expand_sub(grad_tail, scale))
                dx[row : row + ROW_TILE, start:COLS] = pl.cast(result_tail, pl.BF16)
    return dx

@pl.jit
def bf16_wide_scan_pipelined_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.BF16],
    y: pl.Tensor[[ROWS, COLS], pl.BF16],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.BF16]],
):
    for block in pl.spmd(ROWS // LONG_ROW_TILE, name_hint="bf16_pipelined_rows"):
        row = block * LONG_ROW_TILE
        dot = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(LONG_REDUCE_CHUNKS):
            col = chunk * LONG_REDUCE_TILE
            reduce_grad = pl.cast(dy[row : row + ROW_TILE, col : col + LONG_REDUCE_TILE], pl.FP32)
            reduce_saved = pl.cast(y[row : row + ROW_TILE, col : col + LONG_REDUCE_TILE], pl.FP32)
            reduce_partial = pl.reshape(pl.row_sum(pl.mul(reduce_grad, reduce_saved)), [1, ROW_TILE])
            dot = pl.add(dot, reduce_partial)
        if LONG_REDUCE_TAIL > 0:
            start = LONG_REDUCE_CHUNKS * LONG_REDUCE_TILE
            reduce_grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
            reduce_saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
            reduce_partial_tail = pl.reshape(pl.row_sum(pl.mul(reduce_grad_tail, reduce_saved_tail)), [1, ROW_TILE])
            dot = pl.add(dot, reduce_partial_tail)
        scale = pl.reshape(dot, [ROW_TILE, 1])
        for chunk in pl.pipeline(FULL_CHUNKS, stage=PIPELINE_STAGE):
            col = chunk * COL_TILE
            grad = pl.cast(dy[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
            saved = pl.cast(y[row : row + ROW_TILE, col : col + COL_TILE], pl.FP32)
            result = pl.mul(saved, pl.row_expand_sub(grad, scale))
            dx[row : row + ROW_TILE, col : col + COL_TILE] = pl.cast(result, pl.BF16)
        if TAIL_COLS > 0:
            start = FULL_CHUNKS * COL_TILE
            grad_tail = pl.cast(dy[row : row + ROW_TILE, start:COLS], pl.FP32)
            saved_tail = pl.cast(y[row : row + ROW_TILE, start:COLS], pl.FP32)
            result_tail = pl.mul(saved_tail, pl.row_expand_sub(grad_tail, scale))
            dx[row : row + ROW_TILE, start:COLS] = pl.cast(result_tail, pl.BF16)
    return dx

@pl.jit
def fp32_streamed_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Tensor[[ROWS, COLS], pl.FP32],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for row in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp32_streamed_rows"):
            dot = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for chunk in pl.range(FULL_CHUNKS):
                col = chunk * COL_TILE
                grad = dy[row : row + ROW_TILE, col : col + COL_TILE]
                saved = y[row : row + ROW_TILE, col : col + COL_TILE]
                partial = pl.reshape(pl.row_sum(pl.mul(grad, saved)), [1, ROW_TILE])
                dot = pl.add(dot, partial)
            if TAIL_COLS > 0:
                start = FULL_CHUNKS * COL_TILE
                grad_tail = dy[row : row + ROW_TILE, start:COLS]
                saved_tail = y[row : row + ROW_TILE, start:COLS]
                partial_tail = pl.reshape(pl.row_sum(pl.mul(grad_tail, saved_tail)), [1, ROW_TILE])
                dot = pl.add(dot, partial_tail)
            scale = pl.reshape(dot, [ROW_TILE, 1])
            for chunk in pl.range(FULL_CHUNKS):
                col = chunk * COL_TILE
                grad = dy[row : row + ROW_TILE, col : col + COL_TILE]
                saved = y[row : row + ROW_TILE, col : col + COL_TILE]
                dx[row : row + ROW_TILE, col : col + COL_TILE] = pl.mul(
                    saved, pl.row_expand_sub(grad, scale)
                )
            if TAIL_COLS > 0:
                start = FULL_CHUNKS * COL_TILE
                grad_tail = dy[row : row + ROW_TILE, start:COLS]
                saved_tail = y[row : row + ROW_TILE, start:COLS]
                dx[row : row + ROW_TILE, start:COLS] = pl.mul(
                    saved_tail, pl.row_expand_sub(grad_tail, scale)
                )
    return dx


@pl.jit
def bf16_residual_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.BF16],
    y: pl.Tensor[[ROWS, COLS], pl.BF16],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.BF16]],
):
    for row in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bf16_residual_rows"):
            valid_rows = pl.min(ROW_TILE, ROWS - row)
            dot = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            sum_tmp = pl.create_tile([ROW_TILE, COL_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            for chunk in pl.range((COLS + COL_TILE - 1) // COL_TILE):
                col = chunk * COL_TILE
                valid_cols = pl.min(COL_TILE, COLS - col)
                grad_input = pl.load(dy, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                saved_input = pl.load(y, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                grad = pl.cast(grad_input, pl.FP32)
                saved = pl.cast(saved_input, pl.FP32)
                grad = pl.set_validshape(pl.fillpad(grad, pad_value=pl.PadValue.zero), ROW_TILE, COL_TILE)
                saved = pl.set_validshape(pl.fillpad(saved, pad_value=pl.PadValue.zero), ROW_TILE, COL_TILE)
                partial = pl.reshape(pl.row_sum(pl.mul(grad, saved), sum_tmp), [1, ROW_TILE])
                dot = pl.add(dot, partial)
            scale = pl.reshape(dot, [ROW_TILE, 1])
            for chunk in pl.range((COLS + COL_TILE - 1) // COL_TILE):
                col = chunk * COL_TILE
                valid_cols = pl.min(COL_TILE, COLS - col)
                grad_input = pl.load(dy, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                saved_input = pl.load(y, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                grad = pl.cast(grad_input, pl.FP32)
                saved = pl.cast(saved_input, pl.FP32)
                result = pl.mul(saved, pl.row_expand_sub(grad, scale))
                result = pl.set_validshape(result, valid_rows, valid_cols)
                pl.store(pl.cast(result, pl.BF16), [row, col], dx)
    return dx


@pl.jit
def fp32_residual_backward(
    dy: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Tensor[[ROWS, COLS], pl.FP32],
    dx: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for row in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp32_residual_rows"):
            valid_rows = pl.min(ROW_TILE, ROWS - row)
            dot = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            sum_tmp = pl.create_tile([ROW_TILE, COL_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            for chunk in pl.range((COLS + COL_TILE - 1) // COL_TILE):
                col = chunk * COL_TILE
                valid_cols = pl.min(COL_TILE, COLS - col)
                grad_input = pl.load(dy, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                saved_input = pl.load(y, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                grad = grad_input
                saved = saved_input
                grad = pl.set_validshape(pl.fillpad(grad, pad_value=pl.PadValue.zero), ROW_TILE, COL_TILE)
                saved = pl.set_validshape(pl.fillpad(saved, pad_value=pl.PadValue.zero), ROW_TILE, COL_TILE)
                partial = pl.reshape(pl.row_sum(pl.mul(grad, saved), sum_tmp), [1, ROW_TILE])
                dot = pl.add(dot, partial)
            scale = pl.reshape(dot, [ROW_TILE, 1])
            for chunk in pl.range((COLS + COL_TILE - 1) // COL_TILE):
                col = chunk * COL_TILE
                valid_cols = pl.min(COL_TILE, COLS - col)
                grad_input = pl.load(dy, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                saved_input = pl.load(y, [row, col], [ROW_TILE, COL_TILE], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec, clamp=True)
                grad = grad_input
                saved = saved_input
                result = pl.mul(saved, pl.row_expand_sub(grad, scale))
                result = pl.set_validshape(result, valid_rows, valid_cols)
                pl.store(result, [row, col], dx)
    return dx


def golden_backward(tensors):
    import torch

    grad = tensors["dy"].to(dtype=torch.float32)
    saved = tensors["y"].to(dtype=torch.float32)
    dot = (grad * saved).sum(dim=-1, keepdim=True)
    tensors["dx"][:] = (saved * (grad - dot)).to(dtype=tensors["dx"].dtype)


def build(shape):
    import torch
    from golden import TensorSpec

    global ROWS, COLS, ROW_TILE, FULL_CHUNKS, TAIL_COLS
    global LONG_REDUCE_CHUNKS, LONG_REDUCE_TAIL
    ROWS = int(shape["M"])
    COLS = int(shape["N"])
    if ROWS <= 0 or COLS <= 0:
        raise ValueError("M and N must be positive")
    ROW_TILE = REQUESTED_ROW_TILE
    FULL_CHUNKS, TAIL_COLS = divmod(COLS, COL_TILE)
    LONG_REDUCE_CHUNKS, LONG_REDUCE_TAIL = divmod(COLS, LONG_REDUCE_TILE)
    dtype_name = shape["dtype"]
    if dtype_name == "bf16":
        torch_dtype = torch.bfloat16
        fn = bf16_spmd_backward if COLS > 1024 and ROWS * COLS >= 1048576 else bf16_streamed_backward
        rtol = atol = 1e-2
    elif dtype_name == "fp32":
        torch_dtype = torch.float32
        fn = fp32_streamed_backward
        rtol = atol = 1e-4
    else:
        raise ValueError("dtype must be 'fp32' or 'bf16'")
    if ROWS % ROW_TILE != 0 or COLS % 16 != 0 or COLS < COL_TILE:
        ROW_TILE = 8
        fn = bf16_residual_backward if dtype_name == 'bf16' else fp32_residual_backward
    if (dtype_name == 'bf16' and COLS > 1024 and ROWS * COLS >= 1048576
            and ROWS % LONG_ROW_TILE == 0 and TAIL_COLS == 0
            and FULL_CHUNKS >= LONG_MIN_FULL_CHUNKS):
        ROW_TILE = LONG_ROW_TILE
        fn = bf16_pipelined_backward
    if fn is bf16_pipelined_backward and LONG_REDUCE_TAIL == 0:
        fn = bf16_wide_scan_pipelined_backward
    specs = [
        TensorSpec("dy", [ROWS, COLS], torch_dtype, init_value=torch.randn),
        TensorSpec(
            "y",
            [ROWS, COLS],
            torch_dtype,
            init_value=lambda: torch.softmax(torch.randn(ROWS, COLS), dim=-1),
        ),
        TensorSpec("dx", [ROWS, COLS], torch_dtype),
    ]
    return fn, specs, golden_backward, rtol, atol

if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    fn, specs, golden_fn, rtol, atol = build(DEFAULT_SHAPE)
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=rtol,
        atol=atol,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
