# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row-resident FP32 SwiGLU with dynamic per-token INT8 quantization."""

import pypto.language as pl

ROWS = 512
HIDDEN = 1024
ROW_TILE = 8
COL_TILE = 1024
CHUNK_TILE = 256


@pl.jit
def swiglu_quant(
    x: pl.Tensor[[ROWS, 2 * HIDDEN], pl.FP32],
    q: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[1, ROWS], pl.FP32]],
):
    for row_start in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_swiglu_quant"):
            valid_rows = pl.min(ROW_TILE, ROWS - row_start)
            gate = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, 0], valid_shape=[valid_rows, HIDDEN], pad_value=0, clamp=True)
            up = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, HIDDEN], valid_shape=[valid_rows, HIDDEN], pad_value=0, clamp=True)
            denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
            activated = pl.mul(pl.div(gate, denominator), up)
            activated = pl.fillpad(activated, pad_value=0)
            row_amax = pl.reshape(pl.row_max(pl.abs(activated)), [1, ROW_TILE])
            row_scale = pl.mul(row_amax, 1.0 / 127.0)
            row_scale = pl.add(row_scale, pl.cmp(row_scale, 0.0, cmp_type=0))
            row_scale_col = pl.reshape(row_scale, [ROW_TILE, 1])
            normalized = pl.row_expand_div(activated, row_scale_col)
            clamped = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
            rounded = pl.cast(clamped, pl.INT32, mode="rint")
            exact_half = pl.cast(rounded, pl.FP16, mode="rint")
            quantized = pl.cast(exact_half, pl.INT8, mode="rint")
            quantized = pl.set_validshape(quantized, valid_rows, HIDDEN)
            if HIDDEN == 1:
                for row_offset in pl.range(valid_rows):
                    q_value = pl.read(quantized, [row_offset, 0])
                    pl.write(q, [row_start + row_offset, 0], q_value)
            else:
                q = pl.assemble(q, quantized, [row_start, 0])
            scale_tile = pl.set_validshape(row_scale, 1, valid_rows)
            scale = pl.assemble(scale, scale_tile, [0, row_start])
    return q, scale


@pl.jit
def swiglu_quant_bounded(
    x: pl.Tensor[[ROWS, 2 * HIDDEN], pl.FP32],
    q: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.INT8]],
    scale: pl.Out[pl.Tensor[[1, ROWS], pl.FP32]],
):
    for row_start in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="wide_swiglu_quant"):
            valid_rows = pl.min(ROW_TILE, ROWS - row_start)
            row_amax = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, HIDDEN, CHUNK_TILE):
                valid_cols = pl.min(CHUNK_TILE, HIDDEN - col_start)
                gate = pl.slice(x, [ROW_TILE, CHUNK_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols], pad_value=0, clamp=True)
                up = pl.slice(x, [ROW_TILE, CHUNK_TILE], [row_start, HIDDEN + col_start], valid_shape=[valid_rows, valid_cols], pad_value=0, clamp=True)
                denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
                activated = pl.mul(pl.div(gate, denominator), up)
                activated = pl.fillpad(activated, pad_value=0)
                chunk_amax = pl.reshape(pl.row_max(pl.abs(activated)), [1, ROW_TILE])
                row_amax = pl.maximum(row_amax, chunk_amax)
            row_scale = pl.mul(row_amax, 1.0 / 127.0)
            row_scale = pl.add(row_scale, pl.cmp(row_scale, 0.0, cmp_type=0))
            scale_tile = pl.set_validshape(row_scale, 1, valid_rows)
            scale = pl.assemble(scale, scale_tile, [0, row_start])
            row_scale_col = pl.reshape(row_scale, [ROW_TILE, 1])
            for col_start in pl.range(0, HIDDEN, CHUNK_TILE):
                valid_cols = pl.min(CHUNK_TILE, HIDDEN - col_start)
                gate = pl.slice(x, [ROW_TILE, CHUNK_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols], pad_value=0, clamp=True)
                up = pl.slice(x, [ROW_TILE, CHUNK_TILE], [row_start, HIDDEN + col_start], valid_shape=[valid_rows, valid_cols], pad_value=0, clamp=True)
                denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
                activated = pl.mul(pl.div(gate, denominator), up)
                normalized = pl.row_expand_div(activated, row_scale_col)
                clamped = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(clamped, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                quantized = pl.cast(exact_half, pl.INT8, mode="rint")
                quantized = pl.set_validshape(quantized, valid_rows, valid_cols)
                q = pl.assemble(q, quantized, [row_start, col_start])
    return q, scale



def golden_swiglu_quant(tensors):
    import torch

    hidden = tensors["x"].shape[1] // 2
    gate = tensors["x"][:, :hidden].float()
    up = tensors["x"][:, hidden:].float()
    activated = torch.nn.functional.silu(gate) * up
    row_scale = activated.abs().amax(dim=1, keepdim=True) / 127.0
    row_scale = torch.where(row_scale == 0.0, torch.ones_like(row_scale), row_scale)
    tensors["q"][:] = torch.clamp(torch.round(activated / row_scale), -127, 127).to(torch.int8)
    tensors["scale"][:] = row_scale.transpose(0, 1)


def build(shape):
    import torch
    from golden import TensorSpec

    global ROWS, HIDDEN, ROW_TILE, COL_TILE
    ROWS = int(shape["M"])
    HIDDEN = int(shape["N"])
    if ROWS <= 0 or HIDDEN <= 0:
        raise ValueError("M and N must be positive")
    COL_TILE = ((HIDDEN + 31) // 32) * 32
    if COL_TILE <= 1024:
        ROW_TILE = min(8, max(8, (8192 // COL_TILE // 8) * 8))
        kernel = swiglu_quant
    else:
        ROW_TILE = min(8, max(8, (8192 // CHUNK_TILE // 8) * 8))
        kernel = swiglu_quant_bounded
    specs = [
        TensorSpec("x", [ROWS, 2 * HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("q", [ROWS, HIDDEN], torch.int8),
        TensorSpec("scale", [1, ROWS], torch.float32),
    ]
    return kernel, specs, golden_swiglu_quant, 1e-4, 1e-4

DEFAULT_SHAPE = {"M": 512, "N": 1024}

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
