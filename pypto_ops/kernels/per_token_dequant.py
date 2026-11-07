# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-token dequantization — x[m, n] = q[m, n] * scale[m] (int8 codes, per-token fp32 scales, fp32 output)."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024}


TASK_COUNT = 20
WORKERS = 20
ROW_TILE = 32
COL_TILE = 256


@pl.jit
def per_token_dequant(q: pl.Tensor, scale: pl.Tensor, x: pl.Out[pl.Tensor]):
    rows, cols = q.shape
    for worker in pl.spmd(WORKERS, name_hint="balanced_row_owner"):
        row_begin = (worker * rows) // WORKERS
        row_end = ((worker + 1) * rows) // WORKERS
        for row_start in pl.range(row_begin, row_end, ROW_TILE):
            valid_rows = pl.min(ROW_TILE, row_end - row_start)
            scale_tile = pl.slice(scale, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            for col_index in pl.pipeline((cols + COL_TILE - 1) // COL_TILE, stage=2):
                col_start = col_index * COL_TILE
                valid_cols = pl.min(COL_TILE, cols - col_start)
                quant_tile = pl.slice(q, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                half_tile = pl.cast(quant_tile, target_type=pl.FP16)
                float_tile = pl.cast(half_tile, target_type=pl.FP32)
                result_tile = pl.row_expand_mul(float_tile, scale_tile)
                result_valid = pl.set_validshape(result_tile, valid_rows, valid_cols)
                x = pl.assemble(x, result_valid, [row_start, col_start])
    return x


@pl.jit
def per_token_dequant_column(q: pl.Tensor, scale: pl.Tensor, x: pl.Out[pl.Tensor]):
    rows, cols = q.shape
    for worker in pl.spmd(WORKERS, name_hint="balanced_column_owner"):
        row_begin = (worker * rows) // WORKERS
        row_end = ((worker + 1) * rows) // WORKERS
        for row_start in pl.range(row_begin, row_end, ROW_TILE):
            valid_rows = pl.min(ROW_TILE, row_end - row_start)
            quant_tile = pl.slice(q, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            scale_tile = pl.slice(scale, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            half_tile = pl.cast(quant_tile, target_type=pl.FP16)
            float_tile = pl.cast(half_tile, target_type=pl.FP32)
            result_tile = pl.mul(float_tile, scale_tile)
            result_valid = pl.set_validshape(result_tile, valid_rows, 1)
            x = pl.assemble(x, result_valid, [row_start, 0])
    return x


def golden_dequant(tensors):
    tensors["x"][:] = tensors["q"].float() * tensors["scale"]


def build(shape: dict):
    import torch
    from golden import TensorSpec

    global WORKERS, ROW_TILE, COL_TILE
    rows, cols = shape["M"], shape["N"]
    WORKERS = min(TASK_COUNT, rows)
    rows_per_worker = (rows + WORKERS - 1) // WORKERS
    ROW_TILE = min(64, 8 * ((rows_per_worker + 7) // 8))
    COL_TILE = min(256, max(32, 32 * ((cols + 31) // 32)))
    if cols == 1:
        ROW_TILE = max(32, ROW_TILE)
        COL_TILE = 1
    specs = [
        TensorSpec("q", [rows, cols], torch.int8, init_value=lambda: torch.randint(-128, 128, (rows, cols), dtype=torch.int8)),
        TensorSpec("scale", [rows, 1], torch.float32, init_value=torch.randn),
        TensorSpec("x", [rows, cols], torch.float32),
    ]
    kernel = per_token_dequant_column if cols == 1 else per_token_dequant
    return kernel, specs, golden_dequant, 1e-4, 1e-4

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
