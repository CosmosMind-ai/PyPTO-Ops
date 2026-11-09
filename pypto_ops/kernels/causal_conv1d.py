# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Causal depthwise convolution with balanced tasks and input-precision halos."""

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}



M_DIM = pl.dynamic("M_DIM")
N_DIM = pl.dynamic("N_DIM")


@pl.jit
def causal_conv_bf16_c256(
    x: pl.Tensor[[M_DIM, N_DIM], pl.BF16],
    w: pl.Tensor[[N_DIM, 4], pl.BF16],
    y: pl.Out[pl.Tensor[[M_DIM, N_DIM], pl.BF16]],
):
    rows = pl.tensor.dim(x, 0)
    cols = pl.tensor.dim(x, 1)
    row_tiles = rows // 32
    channel_blocks = cols // 256
    strips = (96 + channel_blocks - 1) // channel_blocks
    if strips > row_tiles:
        strips = row_tiles
    flat_w = pl.reshape(w, [1, cols * 4])
    for task in pl.parallel(strips * channel_blocks):
        strip = task // channel_blocks
        col_start = (task % channel_blocks) * 256
        tile_begin = strip * row_tiles // strips
        tile_end = (strip + 1) * row_tiles // strips
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="halo_fused"):
            packed_input = pl.load(flat_w, [0, col_start * 4], [1, 256 * 4])
            packed = pl.tile.cast(packed_input, pl.FP32)
            weight0 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0001)
            weight1 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0010)
            weight2 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0100)
            weight3 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P1000)
            bulk_begin = tile_begin
            if tile_begin == 0:
                row_start = 0
                head_input = pl.load(x, [0, col_start], [32, 256])
                window_head = pl.tile.full([35, 256], dtype=pl.BF16, value=0.0)
                window_head = pl.tile.assemble(window_head, head_input, [3, 0])
                term0_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 256], [0, 0]), pl.FP32), weight0)
                term1_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 256], [1, 0]), pl.FP32), weight1)
                accum_head = pl.tile.add(term0_head, term1_head)
                term2_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 256], [2, 0]), pl.FP32), weight2)
                accum_head = pl.tile.add(accum_head, term2_head)
                term3_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 256], [3, 0]), pl.FP32), weight3)
                accum_head = pl.tile.add(accum_head, term3_head)
                pl.store(pl.tile.cast(accum_head, pl.BF16, mode="rint"), [row_start, col_start], y)
                bulk_begin = 1
            for tile_index in pl.pipeline(bulk_begin, tile_end, stage=2):
                row_start = tile_index * 32
                window_input = pl.load(x, [row_start - 3, col_start], [35, 256])
                window = window_input
                term0 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 256], [0, 0]), pl.FP32), weight0)
                term1 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 256], [1, 0]), pl.FP32), weight1)
                accum = pl.tile.add(term0, term1)
                term2 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 256], [2, 0]), pl.FP32), weight2)
                accum = pl.tile.add(accum, term2)
                term3 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 256], [3, 0]), pl.FP32), weight3)
                accum = pl.tile.add(accum, term3)
                pl.store(pl.tile.cast(accum, pl.BF16, mode="rint"), [row_start, col_start], y)
    return y


@pl.jit
def causal_conv_bf16_c128(
    x: pl.Tensor[[M_DIM, N_DIM], pl.BF16],
    w: pl.Tensor[[N_DIM, 4], pl.BF16],
    y: pl.Out[pl.Tensor[[M_DIM, N_DIM], pl.BF16]],
):
    rows = pl.tensor.dim(x, 0)
    cols = pl.tensor.dim(x, 1)
    row_tiles = rows // 32
    channel_blocks = cols // 128
    strips = (96 + channel_blocks - 1) // channel_blocks
    if strips > row_tiles:
        strips = row_tiles
    flat_w = pl.reshape(w, [1, cols * 4])
    for task in pl.parallel(strips * channel_blocks):
        strip = task // channel_blocks
        col_start = (task % channel_blocks) * 128
        tile_begin = strip * row_tiles // strips
        tile_end = (strip + 1) * row_tiles // strips
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="halo_fused"):
            packed_input = pl.load(flat_w, [0, col_start * 4], [1, 128 * 4])
            packed = pl.tile.cast(packed_input, pl.FP32)
            weight0 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0001)
            weight1 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0010)
            weight2 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0100)
            weight3 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P1000)
            bulk_begin = tile_begin
            if tile_begin == 0:
                row_start = 0
                head_input = pl.load(x, [0, col_start], [32, 128])
                window_head = pl.tile.full([35, 128], dtype=pl.BF16, value=0.0)
                window_head = pl.tile.assemble(window_head, head_input, [3, 0])
                term0_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 128], [0, 0]), pl.FP32), weight0)
                term1_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 128], [1, 0]), pl.FP32), weight1)
                accum_head = pl.tile.add(term0_head, term1_head)
                term2_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 128], [2, 0]), pl.FP32), weight2)
                accum_head = pl.tile.add(accum_head, term2_head)
                term3_head = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window_head, [32, 128], [3, 0]), pl.FP32), weight3)
                accum_head = pl.tile.add(accum_head, term3_head)
                pl.store(pl.tile.cast(accum_head, pl.BF16, mode="rint"), [row_start, col_start], y)
                bulk_begin = 1
            for tile_index in pl.pipeline(bulk_begin, tile_end, stage=2):
                row_start = tile_index * 32
                window_input = pl.load(x, [row_start - 3, col_start], [35, 128])
                window = window_input
                term0 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 128], [0, 0]), pl.FP32), weight0)
                term1 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 128], [1, 0]), pl.FP32), weight1)
                accum = pl.tile.add(term0, term1)
                term2 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 128], [2, 0]), pl.FP32), weight2)
                accum = pl.tile.add(accum, term2)
                term3 = pl.tile.col_expand_mul(pl.tile.cast(pl.tile.slice(window, [32, 128], [3, 0]), pl.FP32), weight3)
                accum = pl.tile.add(accum, term3)
                pl.store(pl.tile.cast(accum, pl.BF16, mode="rint"), [row_start, col_start], y)
    return y


@pl.jit
def causal_conv_fp32_c128(
    x: pl.Tensor[[M_DIM, N_DIM], pl.FP32],
    w: pl.Tensor[[N_DIM, 4], pl.FP32],
    y: pl.Out[pl.Tensor[[M_DIM, N_DIM], pl.FP32]],
):
    rows = pl.tensor.dim(x, 0)
    cols = pl.tensor.dim(x, 1)
    row_tiles = rows // 32
    channel_blocks = cols // 128
    strips = (96 + channel_blocks - 1) // channel_blocks
    if strips > row_tiles:
        strips = row_tiles
    flat_w = pl.reshape(w, [1, cols * 4])
    for task in pl.parallel(strips * channel_blocks):
        strip = task // channel_blocks
        col_start = (task % channel_blocks) * 128
        tile_begin = strip * row_tiles // strips
        tile_end = (strip + 1) * row_tiles // strips
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="halo_fused"):
            packed_input = pl.load(flat_w, [0, col_start * 4], [1, 128 * 4])
            packed = packed_input
            weight0 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0001)
            weight1 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0010)
            weight2 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P0100)
            weight3 = pl.tile.gather_mask(packed, mask_pattern=pl.tile.MaskPattern.P1000)
            bulk_begin = tile_begin
            if tile_begin == 0:
                row_start = 0
                head_input = pl.load(x, [0, col_start], [32, 128])
                window_head = pl.tile.full([35, 128], dtype=pl.FP32, value=0.0)
                window_head = pl.tile.assemble(window_head, head_input, [3, 0])
                term0_head = pl.tile.col_expand_mul(pl.tile.slice(window_head, [32, 128], [0, 0]), weight0)
                term1_head = pl.tile.col_expand_mul(pl.tile.slice(window_head, [32, 128], [1, 0]), weight1)
                accum_head = pl.tile.add(term0_head, term1_head)
                term2_head = pl.tile.col_expand_mul(pl.tile.slice(window_head, [32, 128], [2, 0]), weight2)
                accum_head = pl.tile.add(accum_head, term2_head)
                term3_head = pl.tile.col_expand_mul(pl.tile.slice(window_head, [32, 128], [3, 0]), weight3)
                accum_head = pl.tile.add(accum_head, term3_head)
                pl.store(accum_head, [row_start, col_start], y)
                bulk_begin = 1
            for tile_index in pl.pipeline(bulk_begin, tile_end, stage=2):
                row_start = tile_index * 32
                window_input = pl.load(x, [row_start - 3, col_start], [35, 128])
                window = window_input
                term0 = pl.tile.col_expand_mul(pl.tile.slice(window, [32, 128], [0, 0]), weight0)
                term1 = pl.tile.col_expand_mul(pl.tile.slice(window, [32, 128], [1, 0]), weight1)
                accum = pl.tile.add(term0, term1)
                term2 = pl.tile.col_expand_mul(pl.tile.slice(window, [32, 128], [2, 0]), weight2)
                accum = pl.tile.add(accum, term2)
                term3 = pl.tile.col_expand_mul(pl.tile.slice(window, [32, 128], [3, 0]), weight3)
                accum = pl.tile.add(accum, term3)
                pl.store(accum, [row_start, col_start], y)
    return y


def build(shape):
    import torch
    import torch.nn.functional as functional
    from golden import TensorSpec

    rows = shape["M"]
    cols = shape["N"]
    dtype = shape["dtype"]
    if dtype not in ("bf16", "fp32") or rows <= 0 or cols <= 0 or rows % 64 or cols % 128:
        raise ValueError("Expected positive M divisible by 64, N divisible by 128, and bf16 or fp32")
    if dtype == "bf16":
        storage_dtype = torch.bfloat16
        width = 256 if cols % 256 == 0 else 128
        fn = globals()[f"causal_conv_bf16_c{width}"]
        tolerance = 2e-2
    else:
        storage_dtype = torch.float32
        fn = causal_conv_fp32_c128
        tolerance = 1e-4
    specs = [
        TensorSpec("x", [rows, cols], storage_dtype, init_value=torch.randn),
        TensorSpec("w", [cols, 4], storage_dtype, init_value=torch.randn),
        TensorSpec("y", [rows, cols], storage_dtype),
    ]

    def golden_fn(tensors):
        inputs = functional.pad(tensors["x"].float().transpose(0, 1).unsqueeze(0), (3, 0))
        weights = tensors["w"].float().unsqueeze(1)
        output = functional.conv1d(inputs, weights, groups=cols)
        tensors["y"][:] = output.squeeze(0).transpose(0, 1).to(storage_dtype)

    return fn, specs, golden_fn, tolerance, tolerance

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
