# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 resident gated RMSNorm with independently bounded streamed fallback."""

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "fp32"}


REQUESTED_ROW_TILE = 8
RESIDENT_ELEMENTS_CAP = 8192
FALLBACK_ROW_TILE = 8
FALLBACK_COL_TILE = 512
STREAM_ELEMENTS_CAP = 4096
STREAM_BYTES_CAP = 188416
EPS = 1e-6


RESIDENT_R8_ROW_TILE = 8

@pl.jit
def resident_r8(
    x: pl.Tensor,
    gate: pl.Tensor,
    gamma: pl.Tensor,
    y: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, RESIDENT_R8_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gated_rms_resident"):
            source = x[row_start:row_start + RESIDENT_R8_ROW_TILE, :]
            square_sum = pl.reshape(pl.row_sum(pl.mul(source, source)), [1, RESIDENT_R8_ROW_TILE])
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse = pl.rsqrt(mean_square, high_precision=True)
            inverse = pl.reshape(inverse, [RESIDENT_R8_ROW_TILE, 1])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), gamma[:, :])
            gate_value = gate[row_start:row_start + RESIDENT_R8_ROW_TILE, :]
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_value)), 1.0))
            activated_gate = pl.mul(gate_value, sigmoid)
            y[row_start:row_start + RESIDENT_R8_ROW_TILE, :] = pl.mul(normalized, activated_gate)
    return y


STREAMED_R8_C128_ROW_TILE = 8
STREAMED_R8_C128_COL_TILE = 128

@pl.jit
def streamed_r8_c128(
    x: pl.Tensor,
    gate: pl.Tensor,
    gamma: pl.Tensor,
    y: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, STREAMED_R8_C128_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gated_rms_rows"):
            square_sum = pl.full([1, STREAMED_R8_C128_ROW_TILE], dtype=pl.FP32, value=0.0)
            for column_block in pl.range(hidden // STREAMED_R8_C128_COL_TILE):
                column_start = column_block * STREAMED_R8_C128_COL_TILE
                source = x[row_start:row_start + STREAMED_R8_C128_ROW_TILE, column_start:column_start + STREAMED_R8_C128_COL_TILE]
                partial = pl.row_sum(pl.mul(source, source))
                square_sum = pl.add(square_sum, pl.reshape(partial, [1, STREAMED_R8_C128_ROW_TILE]))
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse = pl.rsqrt(mean_square, high_precision=True)
            inverse = pl.reshape(inverse, [STREAMED_R8_C128_ROW_TILE, 1])
            for column_block in pl.range(hidden // STREAMED_R8_C128_COL_TILE):
                column_start = column_block * STREAMED_R8_C128_COL_TILE
                source = x[row_start:row_start + STREAMED_R8_C128_ROW_TILE, column_start:column_start + STREAMED_R8_C128_COL_TILE]
                weights = gamma[:, column_start:column_start + STREAMED_R8_C128_COL_TILE]
                normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weights)
                gate_value = gate[row_start:row_start + STREAMED_R8_C128_ROW_TILE, column_start:column_start + STREAMED_R8_C128_COL_TILE]
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_value)), 1.0))
                activated_gate = pl.mul(gate_value, sigmoid)
                y[row_start:row_start + STREAMED_R8_C128_ROW_TILE, column_start:column_start + STREAMED_R8_C128_COL_TILE] = pl.mul(normalized, activated_gate)
    return y


STREAMED_R8_C256_ROW_TILE = 8
STREAMED_R8_C256_COL_TILE = 256

@pl.jit
def streamed_r8_c256(
    x: pl.Tensor,
    gate: pl.Tensor,
    gamma: pl.Tensor,
    y: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, STREAMED_R8_C256_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gated_rms_rows"):
            square_sum = pl.full([1, STREAMED_R8_C256_ROW_TILE], dtype=pl.FP32, value=0.0)
            for column_block in pl.range(hidden // STREAMED_R8_C256_COL_TILE):
                column_start = column_block * STREAMED_R8_C256_COL_TILE
                source = x[row_start:row_start + STREAMED_R8_C256_ROW_TILE, column_start:column_start + STREAMED_R8_C256_COL_TILE]
                partial = pl.row_sum(pl.mul(source, source))
                square_sum = pl.add(square_sum, pl.reshape(partial, [1, STREAMED_R8_C256_ROW_TILE]))
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse = pl.rsqrt(mean_square, high_precision=True)
            inverse = pl.reshape(inverse, [STREAMED_R8_C256_ROW_TILE, 1])
            for column_block in pl.range(hidden // STREAMED_R8_C256_COL_TILE):
                column_start = column_block * STREAMED_R8_C256_COL_TILE
                source = x[row_start:row_start + STREAMED_R8_C256_ROW_TILE, column_start:column_start + STREAMED_R8_C256_COL_TILE]
                weights = gamma[:, column_start:column_start + STREAMED_R8_C256_COL_TILE]
                normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weights)
                gate_value = gate[row_start:row_start + STREAMED_R8_C256_ROW_TILE, column_start:column_start + STREAMED_R8_C256_COL_TILE]
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_value)), 1.0))
                activated_gate = pl.mul(gate_value, sigmoid)
                y[row_start:row_start + STREAMED_R8_C256_ROW_TILE, column_start:column_start + STREAMED_R8_C256_COL_TILE] = pl.mul(normalized, activated_gate)
    return y


STREAMED_R8_C512_ROW_TILE = 8
STREAMED_R8_C512_COL_TILE = 512

@pl.jit
def streamed_r8_c512(
    x: pl.Tensor,
    gate: pl.Tensor,
    gamma: pl.Tensor,
    y: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, STREAMED_R8_C512_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gated_rms_rows"):
            square_sum = pl.full([1, STREAMED_R8_C512_ROW_TILE], dtype=pl.FP32, value=0.0)
            for column_block in pl.range(hidden // STREAMED_R8_C512_COL_TILE):
                column_start = column_block * STREAMED_R8_C512_COL_TILE
                source = x[row_start:row_start + STREAMED_R8_C512_ROW_TILE, column_start:column_start + STREAMED_R8_C512_COL_TILE]
                partial = pl.row_sum(pl.mul(source, source))
                square_sum = pl.add(square_sum, pl.reshape(partial, [1, STREAMED_R8_C512_ROW_TILE]))
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse = pl.rsqrt(mean_square, high_precision=True)
            inverse = pl.reshape(inverse, [STREAMED_R8_C512_ROW_TILE, 1])
            for column_block in pl.range(hidden // STREAMED_R8_C512_COL_TILE):
                column_start = column_block * STREAMED_R8_C512_COL_TILE
                source = x[row_start:row_start + STREAMED_R8_C512_ROW_TILE, column_start:column_start + STREAMED_R8_C512_COL_TILE]
                weights = gamma[:, column_start:column_start + STREAMED_R8_C512_COL_TILE]
                normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weights)
                gate_value = gate[row_start:row_start + STREAMED_R8_C512_ROW_TILE, column_start:column_start + STREAMED_R8_C512_COL_TILE]
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_value)), 1.0))
                activated_gate = pl.mul(gate_value, sigmoid)
                y[row_start:row_start + STREAMED_R8_C512_ROW_TILE, column_start:column_start + STREAMED_R8_C512_COL_TILE] = pl.mul(normalized, activated_gate)
    return y



RESIDENT_KERNELS = {
    8: resident_r8,
}
STREAM_KERNELS = {
    128: streamed_r8_c128,
    256: streamed_r8_c256,
    512: streamed_r8_c512,
}


def geometry(shape):
    rows, hidden = shape["M"], shape["N"]
    if shape["dtype"] != "fp32" or rows <= 0 or hidden <= 0:
        raise ValueError("Expected positive FP32 dimensions")
    if rows % 64 or hidden % 128:
        raise ValueError("Expected M divisible by 64 and N divisible by 128")
    row_tile = REQUESTED_ROW_TILE
    while row_tile > 8 and (rows % row_tile or row_tile * hidden > RESIDENT_ELEMENTS_CAP):
        row_tile //= 2
    if rows % row_tile == 0 and row_tile * hidden <= RESIDENT_ELEMENTS_CAP:
        return "resident", row_tile, hidden
    row_tile = FALLBACK_ROW_TILE
    col_tile = FALLBACK_COL_TILE
    while hidden % col_tile:
        col_tile //= 2
    while row_tile * col_tile > STREAM_ELEMENTS_CAP:
        col_tile //= 2
    estimated_bytes = 40 * row_tile * col_tile + 4 * col_tile + 64 * row_tile + 16384
    if rows % row_tile or col_tile < 128 or estimated_bytes > STREAM_BYTES_CAP:
        raise ValueError("Fallback geometry exceeds streamed admission bounds")
    return "streamed", row_tile, col_tile


def golden_gated_rms(tensors):
    import torch

    source = tensors["x"]
    inverse = torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + EPS)
    gate = tensors["gate"]
    tensors["y"][:] = (source * inverse * tensors["gamma"]) * (gate * torch.sigmoid(gate))


def build(shape):
    import torch
    from golden import TensorSpec

    path, row_tile, col_tile = geometry(shape)
    selected = RESIDENT_KERNELS[row_tile] if path == "resident" else STREAM_KERNELS[col_tile]
    rows, hidden = shape["M"], shape["N"]
    specs = [
        TensorSpec("x", [rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gate", [rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], torch.float32),
    ]
    return selected, specs, golden_gated_rms, 1e-4, 1e-4

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
