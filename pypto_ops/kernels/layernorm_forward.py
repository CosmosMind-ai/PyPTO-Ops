# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LayerNorm forward — storage-dtype I/O with FP32 statistics."""
import pypto.language as pl

REQUESTED_CHUNK = 2 ** 9
CACHE_ENABLED = 1
RESIDENT_BUDGET_BYTES = 1024 * 192
ROW_TILE = 8
CHUNK_TILE = 256
PADDED_N = 4096
OUTPUT_DTYPE = pl.BF16
EPS = 1e-5


@pl.jit
def compact_layer_norm(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    block_count = (rows + ROW_TILE - 1) // ROW_TILE
    for block in pl.spmd(block_count, name_hint="compact_native_centered_rows"):
        row = block * ROW_TILE
        valid_rows = pl.min(ROW_TILE, rows - row)
        native_cache = pl.load(
            x, [row, 0], [ROW_TILE, PADDED_N],
            valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec,
        )
        reduction_scratch = pl.tile.create([ROW_TILE, CHUNK_TILE], dtype=pl.FP32)
        zero_sum = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (sum_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_sum,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.slice(
                native_cache, [ROW_TILE, CHUNK_TILE], [0, column],
                valid_shape=[valid_rows, valid_columns],
            )
            tile_x = pl.cast(tile_x, target_type=pl.FP32)
            tile_x = pl.fillpad(tile_x, 0.0)
            chunk_sum = pl.row_sum(tile_x, tmp_tile=reduction_scratch)
            chunk_sum = pl.reshape(chunk_sum, [1, ROW_TILE])
            next_sum = pl.add(sum_state, chunk_sum)
            (row_sum,) = pl.yield_(next_sum)
        mean_row = pl.mul(row_sum, 1.0 / hidden)
        mean = pl.reshape(mean_row, [ROW_TILE, 1])
        zero_variance = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (variance_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_variance,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.slice(
                native_cache, [ROW_TILE, CHUNK_TILE], [0, column],
                valid_shape=[valid_rows, valid_columns],
            )
            tile_x = pl.cast(tile_x, target_type=pl.FP32)
            centered = pl.row_expand_sub(tile_x, mean)
            centered = pl.set_validshape(centered, valid_rows, valid_columns)
            centered = pl.fillpad(centered, 0.0)
            squares = pl.mul(centered, centered)
            chunk_variance = pl.row_sum(squares, tmp_tile=reduction_scratch)
            chunk_variance = pl.reshape(chunk_variance, [1, ROW_TILE])
            next_variance = pl.add(variance_state, chunk_variance)
            (row_m2,) = pl.yield_(next_variance)
        variance_eps = pl.add(pl.mul(row_m2, 1.0 / hidden), EPS)
        inverse_std = pl.rsqrt(variance_eps)
        inverse_square = pl.mul(inverse_std, inverse_std)
        residual = pl.mul(variance_eps, inverse_square)
        correction = pl.add(pl.mul(residual, -0.5), 1.5)
        inverse_std = pl.mul(inverse_std, correction)
        inverse_std = pl.reshape(inverse_std, [ROW_TILE, 1])
        for column in pl.range(0, hidden, CHUNK_TILE):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.slice(
                native_cache, [ROW_TILE, CHUNK_TILE], [0, column],
                valid_shape=[valid_rows, valid_columns],
            )
            tile_x = pl.cast(tile_x, target_type=pl.FP32)
            centered = pl.row_expand_sub(tile_x, mean)
            normalized = pl.row_expand_mul(centered, inverse_std)
            normalized = pl.cast(normalized, target_type=OUTPUT_DTYPE, mode="rint")
            normalized = pl.set_validshape(normalized, valid_rows, valid_columns)
            y = pl.store(normalized, [row, column], y)
    return y


@pl.jit
def streaming_layer_norm(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    block_count = (rows + ROW_TILE - 1) // ROW_TILE
    for block in pl.spmd(block_count, name_hint="streaming_centered_rows"):
        row = block * ROW_TILE
        valid_rows = pl.min(ROW_TILE, rows - row)
        reduction_scratch = pl.tile.create([ROW_TILE, CHUNK_TILE], dtype=pl.FP32)
        zero_sum = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (sum_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_sum,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.load(
                x, [row, column], [ROW_TILE, CHUNK_TILE],
                valid_shape=[valid_rows, valid_columns], target_memory=pl.MemorySpace.Vec,
            )
            tile_x = pl.cast(tile_x, target_type=pl.FP32)
            tile_x = pl.fillpad(tile_x, 0.0)
            chunk_sum = pl.row_sum(tile_x, tmp_tile=reduction_scratch)
            chunk_sum = pl.reshape(chunk_sum, [1, ROW_TILE])
            next_sum = pl.add(sum_state, chunk_sum)
            (row_sum,) = pl.yield_(next_sum)
        mean_row = pl.mul(row_sum, 1.0 / hidden)
        mean = pl.reshape(mean_row, [ROW_TILE, 1])
        zero_variance = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (variance_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_variance,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.load(
                x, [row, column], [ROW_TILE, CHUNK_TILE],
                valid_shape=[valid_rows, valid_columns], target_memory=pl.MemorySpace.Vec,
            )
            tile_x = pl.cast(tile_x, target_type=pl.FP32)
            centered = pl.row_expand_sub(tile_x, mean)
            centered = pl.set_validshape(centered, valid_rows, valid_columns)
            centered = pl.fillpad(centered, 0.0)
            squares = pl.mul(centered, centered)
            chunk_variance = pl.row_sum(squares, tmp_tile=reduction_scratch)
            chunk_variance = pl.reshape(chunk_variance, [1, ROW_TILE])
            next_variance = pl.add(variance_state, chunk_variance)
            (row_m2,) = pl.yield_(next_variance)
        variance_eps = pl.add(pl.mul(row_m2, 1.0 / hidden), EPS)
        inverse_std = pl.rsqrt(variance_eps)
        inverse_square = pl.mul(inverse_std, inverse_std)
        residual = pl.mul(variance_eps, inverse_square)
        correction = pl.add(pl.mul(residual, -0.5), 1.5)
        inverse_std = pl.mul(inverse_std, correction)
        inverse_std = pl.reshape(inverse_std, [ROW_TILE, 1])
        for column in pl.range(0, hidden, CHUNK_TILE):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.load(
                x, [row, column], [ROW_TILE, CHUNK_TILE],
                valid_shape=[valid_rows, valid_columns], target_memory=pl.MemorySpace.Vec,
            )
            tile_x = pl.cast(tile_x, target_type=pl.FP32)
            centered = pl.row_expand_sub(tile_x, mean)
            normalized = pl.row_expand_mul(centered, inverse_std)
            normalized = pl.cast(normalized, target_type=OUTPUT_DTYPE, mode="rint")
            normalized = pl.set_validshape(normalized, valid_rows, valid_columns)
            y = pl.store(normalized, [row, column], y)
    return y





@pl.jit
def compact_layer_norm_fp32(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    block_count = (rows + ROW_TILE - 1) // ROW_TILE
    for block in pl.spmd(block_count, name_hint="compact_native_centered_rows"):
        row = block * ROW_TILE
        valid_rows = pl.min(ROW_TILE, rows - row)
        native_cache = pl.load(
            x, [row, 0], [ROW_TILE, PADDED_N],
            valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec,
        )
        reduction_scratch = pl.tile.create([ROW_TILE, CHUNK_TILE], dtype=pl.FP32)
        zero_sum = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (sum_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_sum,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.slice(
                native_cache, [ROW_TILE, CHUNK_TILE], [0, column],
                valid_shape=[valid_rows, valid_columns],
            )
            tile_x = pl.add(tile_x, 0.0)
            tile_x = pl.fillpad(tile_x, 0.0)
            chunk_sum = pl.row_sum(tile_x, tmp_tile=reduction_scratch)
            chunk_sum = pl.reshape(chunk_sum, [1, ROW_TILE])
            next_sum = pl.add(sum_state, chunk_sum)
            (row_sum,) = pl.yield_(next_sum)
        mean_row = pl.mul(row_sum, 1.0 / hidden)
        mean = pl.reshape(mean_row, [ROW_TILE, 1])
        zero_variance = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (variance_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_variance,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.slice(
                native_cache, [ROW_TILE, CHUNK_TILE], [0, column],
                valid_shape=[valid_rows, valid_columns],
            )
            tile_x = pl.add(tile_x, 0.0)
            centered = pl.row_expand_sub(tile_x, mean)
            centered = pl.set_validshape(centered, valid_rows, valid_columns)
            centered = pl.fillpad(centered, 0.0)
            squares = pl.mul(centered, centered)
            chunk_variance = pl.row_sum(squares, tmp_tile=reduction_scratch)
            chunk_variance = pl.reshape(chunk_variance, [1, ROW_TILE])
            next_variance = pl.add(variance_state, chunk_variance)
            (row_m2,) = pl.yield_(next_variance)
        variance_eps = pl.add(pl.mul(row_m2, 1.0 / hidden), EPS)
        inverse_std = pl.rsqrt(variance_eps)
        inverse_square = pl.mul(inverse_std, inverse_std)
        residual = pl.mul(variance_eps, inverse_square)
        correction = pl.add(pl.mul(residual, -0.5), 1.5)
        inverse_std = pl.mul(inverse_std, correction)
        inverse_std = pl.reshape(inverse_std, [ROW_TILE, 1])
        for column in pl.range(0, hidden, CHUNK_TILE):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.slice(
                native_cache, [ROW_TILE, CHUNK_TILE], [0, column],
                valid_shape=[valid_rows, valid_columns],
            )
            tile_x = pl.add(tile_x, 0.0)
            centered = pl.row_expand_sub(tile_x, mean)
            normalized = pl.row_expand_mul(centered, inverse_std)
            normalized = pl.set_validshape(normalized, valid_rows, valid_columns)
            y = pl.store(normalized, [row, column], y)
    return y


@pl.jit
def streaming_layer_norm_fp32(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    block_count = (rows + ROW_TILE - 1) // ROW_TILE
    for block in pl.spmd(block_count, name_hint="streaming_centered_rows"):
        row = block * ROW_TILE
        valid_rows = pl.min(ROW_TILE, rows - row)
        reduction_scratch = pl.tile.create([ROW_TILE, CHUNK_TILE], dtype=pl.FP32)
        zero_sum = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (sum_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_sum,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.load(
                x, [row, column], [ROW_TILE, CHUNK_TILE],
                valid_shape=[valid_rows, valid_columns], target_memory=pl.MemorySpace.Vec,
            )
            tile_x = pl.add(tile_x, 0.0)
            tile_x = pl.fillpad(tile_x, 0.0)
            chunk_sum = pl.row_sum(tile_x, tmp_tile=reduction_scratch)
            chunk_sum = pl.reshape(chunk_sum, [1, ROW_TILE])
            next_sum = pl.add(sum_state, chunk_sum)
            (row_sum,) = pl.yield_(next_sum)
        mean_row = pl.mul(row_sum, 1.0 / hidden)
        mean = pl.reshape(mean_row, [ROW_TILE, 1])
        zero_variance = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for column, (variance_state,) in pl.range(
            0, hidden, CHUNK_TILE, init_values=(zero_variance,),
        ):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.load(
                x, [row, column], [ROW_TILE, CHUNK_TILE],
                valid_shape=[valid_rows, valid_columns], target_memory=pl.MemorySpace.Vec,
            )
            tile_x = pl.add(tile_x, 0.0)
            centered = pl.row_expand_sub(tile_x, mean)
            centered = pl.set_validshape(centered, valid_rows, valid_columns)
            centered = pl.fillpad(centered, 0.0)
            squares = pl.mul(centered, centered)
            chunk_variance = pl.row_sum(squares, tmp_tile=reduction_scratch)
            chunk_variance = pl.reshape(chunk_variance, [1, ROW_TILE])
            next_variance = pl.add(variance_state, chunk_variance)
            (row_m2,) = pl.yield_(next_variance)
        variance_eps = pl.add(pl.mul(row_m2, 1.0 / hidden), EPS)
        inverse_std = pl.rsqrt(variance_eps)
        inverse_square = pl.mul(inverse_std, inverse_std)
        residual = pl.mul(variance_eps, inverse_square)
        correction = pl.add(pl.mul(residual, -0.5), 1.5)
        inverse_std = pl.mul(inverse_std, correction)
        inverse_std = pl.reshape(inverse_std, [ROW_TILE, 1])
        for column in pl.range(0, hidden, CHUNK_TILE):
            valid_columns = pl.min(CHUNK_TILE, hidden - column)
            tile_x = pl.load(
                x, [row, column], [ROW_TILE, CHUNK_TILE],
                valid_shape=[valid_rows, valid_columns], target_memory=pl.MemorySpace.Vec,
            )
            tile_x = pl.add(tile_x, 0.0)
            centered = pl.row_expand_sub(tile_x, mean)
            normalized = pl.row_expand_mul(centered, inverse_std)
            normalized = pl.set_validshape(normalized, valid_rows, valid_columns)
            y = pl.store(normalized, [row, column], y)
    return y







def golden_layer_norm(tensors):
    import torch

    inputs = tensors["x"].float()
    mean = inputs.mean(dim=-1, keepdim=True)
    variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) * torch.rsqrt(variance + EPS)
    tensors["y"][:] = normalized.to(tensors["y"].dtype)


def build(shape):
    import torch
    from golden import TensorSpec

    global PADDED_N, CHUNK_TILE, OUTPUT_DTYPE
    rows, hidden = int(shape["M"]), int(shape["N"])
    dtype_name = shape.get("dtype", "fp32")
    if rows < 1 or hidden < 1 or dtype_name not in ("fp32", "bf16"):
        raise ValueError("Unsupported shape or dtype")
    tensor_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    OUTPUT_DTYPE = pl.BF16 if dtype_name == "bf16" else pl.FP32
    CHUNK_TILE = min(REQUESTED_CHUNK, ((hidden + 15) // 16) * 16)
    PADDED_N = ((hidden + CHUNK_TILE - 1) // CHUNK_TILE) * CHUNK_TILE
    native_bytes = 2 if dtype_name == "bf16" else 4
    estimated_live_bytes = ROW_TILE * PADDED_N * native_bytes + 8 * ROW_TILE * CHUNK_TILE + 8192
    if CACHE_ENABLED and estimated_live_bytes <= RESIDENT_BUDGET_BYTES:
        fn = compact_layer_norm if dtype_name == "bf16" else compact_layer_norm_fp32
    else:
        fn = streaming_layer_norm if dtype_name == "bf16" else streaming_layer_norm_fp32
    specs = [
        TensorSpec("x", [rows, hidden], tensor_dtype, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], tensor_dtype),
    ]
    tolerance = 2e-2 if dtype_name == "bf16" else 1e-4
    return fn, specs, golden_layer_norm, tolerance, tolerance

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}

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
