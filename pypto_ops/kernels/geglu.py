# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GEGLU — y = g * gelu(u) (tanh approximation), bf16 I/O with fp32 arithmetic; store-overlap family geglu_store_overlap_v1."""
import math

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
ROW_TILE = 7
COL_TILE = 1024
TARGET_TASKS = 128


@pl.jit
def geglu_bf16(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    M = g.shape[0]
    N = g.shape[1]
    col_blocks = (N + COL_TILE - 1) // COL_TILE
    row_bands = (M + ROW_TILE - 1) // ROW_TILE
    task_count = pl.min(pl.min(TARGET_TASKS, (M + 19) // 20), row_bands * col_blocks)
    for task_id in pl.parallel(task_count):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="geglu_tile_stream"):
            total_tiles = row_bands * col_blocks
            tile_begin = task_id * total_tiles // task_count
            tile_end = (task_id + 1) * total_tiles // task_count
            for tile_id in pl.pipeline(tile_begin, tile_end, stage=2):
                row_start = (tile_id // col_blocks) * ROW_TILE
                col_start = (tile_id % col_blocks) * COL_TILE
                valid_rows = pl.min(ROW_TILE, M - row_start)
                valid_cols = pl.min(COL_TILE, N - col_start)
                gate_tile = pl.slice(g, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                up_tile = pl.slice(u, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                work_g = pl.cast(gate_tile, target_type=pl.FP32)
                work_u = pl.cast(up_tile, target_type=pl.FP32)
                g3 = pl.mul(pl.mul(work_g, work_g), work_g)
                poly = pl.mul(pl.add(work_g, pl.mul(g3, 0.044715)), SQRT_2_OVER_PI)
                denominator = pl.add(pl.exp(pl.mul(poly, -2.0)), 1.0)
                gelu = pl.div(work_g, denominator)
                result = pl.mul(gelu, work_u)
                result = pl.cast(result, target_type=pl.BF16, mode="rint")
                y[row_start : row_start + ROW_TILE, col_start : col_start + COL_TILE] = pl.set_validshape(result, valid_rows, valid_cols)
    return y


@pl.jit
def geglu_fp32(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    M = g.shape[0]
    N = g.shape[1]
    col_blocks = (N + COL_TILE - 1) // COL_TILE
    row_bands = (M + ROW_TILE - 1) // ROW_TILE
    task_count = pl.min(pl.min(TARGET_TASKS, (M + 19) // 20), row_bands * col_blocks)
    for task_id in pl.parallel(task_count):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="geglu_tile_stream"):
            total_tiles = row_bands * col_blocks
            tile_begin = task_id * total_tiles // task_count
            tile_end = (task_id + 1) * total_tiles // task_count
            for tile_id in pl.range(tile_begin, tile_end):
                row_start = (tile_id // col_blocks) * ROW_TILE
                col_start = (tile_id % col_blocks) * COL_TILE
                valid_rows = pl.min(ROW_TILE, M - row_start)
                valid_cols = pl.min(COL_TILE, N - col_start)
                gate_tile = pl.slice(g, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                up_tile = pl.slice(u, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                work_g = gate_tile
                work_u = up_tile
                g3 = pl.mul(pl.mul(work_g, work_g), work_g)
                poly = pl.mul(pl.add(work_g, pl.mul(g3, 0.044715)), SQRT_2_OVER_PI)
                denominator = pl.add(pl.exp(pl.mul(poly, -2.0)), 1.0)
                gelu = pl.div(work_g, denominator)
                result = pl.mul(gelu, work_u)
                y[row_start : row_start + ROW_TILE, col_start : col_start + COL_TILE] = pl.set_validshape(result, valid_rows, valid_cols)
    return y


def build(shape):
    import torch
    from golden import TensorSpec

    M, N = shape["M"], shape["N"]
    dtype_name = shape.get("dtype", "fp32")
    torch_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    fn = geglu_bf16 if dtype_name == "bf16" else geglu_fp32
    specs = [
        TensorSpec("g", [M, N], torch_dtype, init_value=torch.randn),
        TensorSpec("u", [M, N], torch_dtype, init_value=torch.randn),
        TensorSpec("y", [M, N], torch_dtype),
    ]

    def golden(tensors):
        gate = tensors["g"].float()
        up = tensors["u"].float()
        result = 0.5 * gate * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate * gate * gate))) * up
        tensors["y"][:] = result if dtype_name == "fp32" else result.to(torch.bfloat16)

    tolerance = 1e-4 if dtype_name == "fp32" else 2e-2
    return fn, specs, golden, tolerance, tolerance

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
