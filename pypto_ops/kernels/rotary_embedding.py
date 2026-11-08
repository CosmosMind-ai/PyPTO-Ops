# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rotary position embedding (rotate-half) — y[:, :H2] = x[:, :H2] * cos - x[:, H2:] * sin; y[:, H2:] = x[:, H2:] * cos + x[:, :H2] * sin. fp32."""
import math

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "fp32"}


ROWS = 512
WIDTH = 1024
HALF = WIDTH // 2
ROW_TILE = 8
COL_TILE = 128
BULK_ROWS = ROWS // ROW_TILE * ROW_TILE
TAIL_ROWS = 1


@pl.jit
def paired_rope(
    x: pl.Tensor[[ROWS, WIDTH], pl.FP32],
    cos: pl.Tensor[[ROWS, HALF], pl.FP32],
    sin: pl.Tensor[[ROWS, HALF], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, WIDTH], pl.FP32]],
):
    for row in pl.parallel(0, BULK_ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paired_rows"):
            for column_block in pl.pipeline(HALF // COL_TILE, stage=2):
                column = column_block * COL_TILE
                lower = x[row:row + ROW_TILE, column:column + COL_TILE]
                upper = x[row:row + ROW_TILE, HALF + column:HALF + column + COL_TILE]
                cosine = cos[row:row + ROW_TILE, column:column + COL_TILE]
                sine = sin[row:row + ROW_TILE, column:column + COL_TILE]
                lower_cos = pl.mul(lower, cosine)
                upper_sin = pl.mul(upper, sine)
                lower_out = pl.sub(lower_cos, upper_sin)
                upper_cos = pl.mul(upper, cosine)
                lower_sin = pl.mul(lower, sine)
                upper_out = pl.add(upper_cos, lower_sin)
                y[row:row + ROW_TILE, column:column + COL_TILE] = lower_out
                y[row:row + ROW_TILE, HALF + column:HALF + column + COL_TILE] = upper_out
    return y


@pl.jit
def paired_rope_with_tail(
    x: pl.Tensor[[ROWS, WIDTH], pl.FP32],
    cos: pl.Tensor[[ROWS, HALF], pl.FP32],
    sin: pl.Tensor[[ROWS, HALF], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, WIDTH], pl.FP32]],
):
    for row in pl.parallel(0, BULK_ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paired_rows"):
            for column_block in pl.pipeline(HALF // COL_TILE, stage=2):
                column = column_block * COL_TILE
                lower = x[row:row + ROW_TILE, column:column + COL_TILE]
                upper = x[row:row + ROW_TILE, HALF + column:HALF + column + COL_TILE]
                cosine = cos[row:row + ROW_TILE, column:column + COL_TILE]
                sine = sin[row:row + ROW_TILE, column:column + COL_TILE]
                lower_cos = pl.mul(lower, cosine)
                upper_sin = pl.mul(upper, sine)
                lower_out = pl.sub(lower_cos, upper_sin)
                upper_cos = pl.mul(upper, cosine)
                lower_sin = pl.mul(lower, sine)
                upper_out = pl.add(upper_cos, lower_sin)
                y[row:row + ROW_TILE, column:column + COL_TILE] = lower_out
                y[row:row + ROW_TILE, HALF + column:HALF + column + COL_TILE] = upper_out
    for tail_row in pl.parallel(BULK_ROWS, ROWS, TAIL_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="paired_tail"):
            for tail_column_block in pl.pipeline(HALF // COL_TILE, stage=2):
                tail_column = tail_column_block * COL_TILE
                tail_lower = x[tail_row:tail_row + TAIL_ROWS, tail_column:tail_column + COL_TILE]
                tail_upper = x[tail_row:tail_row + TAIL_ROWS, HALF + tail_column:HALF + tail_column + COL_TILE]
                tail_cosine = cos[tail_row:tail_row + TAIL_ROWS, tail_column:tail_column + COL_TILE]
                tail_sine = sin[tail_row:tail_row + TAIL_ROWS, tail_column:tail_column + COL_TILE]
                tail_lower_cos = pl.mul(tail_lower, tail_cosine)
                tail_upper_sin = pl.mul(tail_upper, tail_sine)
                tail_lower_out = pl.sub(tail_lower_cos, tail_upper_sin)
                tail_upper_cos = pl.mul(tail_upper, tail_cosine)
                tail_lower_sin = pl.mul(tail_lower, tail_sine)
                tail_upper_out = pl.add(tail_upper_cos, tail_lower_sin)
                y[tail_row:tail_row + TAIL_ROWS, tail_column:tail_column + COL_TILE] = tail_lower_out
                y[tail_row:tail_row + TAIL_ROWS, HALF + tail_column:HALF + tail_column + COL_TILE] = tail_upper_out
    return y


def golden_rope(tensors):
    import torch

    source = tensors["x"]
    cosine = tensors["cos"]
    sine = tensors["sin"]
    half = source.shape[1] // 2
    tensors["y"][:] = torch.cat(
        (
            source[:, :half] * cosine - source[:, half:] * sine,
            source[:, half:] * cosine + source[:, :half] * sine,
        ),
        dim=1,
    )


def build(shape):
    import torch
    from golden import TensorSpec

    global ROWS, WIDTH, HALF, ROW_TILE, COL_TILE, BULK_ROWS, TAIL_ROWS
    assert shape["dtype"] == "fp32"
    ROWS = int(shape["M"])
    WIDTH = int(shape["N"])
    assert ROWS > 0 and WIDTH > 0 and WIDTH % 128 == 0
    HALF = WIDTH // 2
    ROW_TILE = min(ROWS, 8)
    COL_TILE = math.gcd(HALF, 128)
    BULK_ROWS = ROWS // ROW_TILE * ROW_TILE
    TAIL_ROWS = max(1, ROWS - BULK_ROWS)
    specs = [
        TensorSpec("x", [ROWS, WIDTH], torch.float32, init_value=torch.randn),
        TensorSpec("cos", [ROWS, HALF], torch.float32, init_value=torch.randn),
        TensorSpec("sin", [ROWS, HALF], torch.float32, init_value=torch.randn),
        TensorSpec("y", [ROWS, WIDTH], torch.float32),
    ]
    kernel = paired_rope_with_tail if ROWS != BULK_ROWS else paired_rope
    return kernel, specs, golden_rope, 1e-4, 1e-4

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
