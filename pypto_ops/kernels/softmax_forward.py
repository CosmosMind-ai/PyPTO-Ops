# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stable mixed-precision softmax with aligned contiguous maximum graphs."""

import pypto.language as pl

MAX_BATCH = 0
REQUESTED_MAX_SEGMENTS = 16
EPILOGUE_MODE = 0
ROW_BATCH_LOG2 = 2
INPUT_BYPASS = 1
REQUESTED_ROWS = 4
REQUESTED_STAGE = 2
PERSISTENT_BLOCKS = 96
OWNERSHIP_MODE = 1
REDUCTION_MODE = 2
MAX_LAYOUT = 0
SUM_LAYOUT = 2
REQUESTED_LANE_WIDTH = 256
COLUMN_HEIGHT_LIMIT = 64
RESIDENT_ELEMENTS = 16384
MAX_RESIDENT_COLUMNS = 8192
WORKING_SET_BUDGET_BYTES = 196608
FALLBACK_LANES = 256


@pl.jit
def softmax_resident_fp32_1_s0(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_fp32_1_s0"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_fp32_1_s2(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_fp32_1_s2"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s0_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s0_w64_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s2_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s2_w64_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s0_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s0_w64_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s2_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s2_w64_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s0_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s0_w128_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s2_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s2_w128_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s0_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s0_w128_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s2_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s2_w128_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s0_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s0_w256_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s2_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s2_w256_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s0_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s0_w256_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_1_s2_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_1_s2_w256_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_fp32_2_s0(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_fp32_2_s0"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_fp32_2_s2(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_fp32_2_s2"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s0_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s0_w64_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s2_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s2_w64_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s0_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s0_w64_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s2_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s2_w64_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s0_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s0_w128_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s2_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s2_w128_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s0_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s0_w128_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s2_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s2_w128_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s0_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s0_w256_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s2_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s2_w256_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s0_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s0_w256_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_2_s2_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_2_s2_w256_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_fp32_4_s0(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_fp32_4_s0"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_fp32_4_s2(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_fp32_4_s2"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s0_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s0_w64_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s2_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s2_w64_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s0_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s0_w64_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s2_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s2_w64_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s0_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s0_w128_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s2_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s2_w128_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s0_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s0_w128_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s2_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s2_w128_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s0_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s0_w256_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s2_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s2_w256_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s0_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s0_w256_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_fp32_4_s2_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_fp32_4_s2_w256_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )

            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
    
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_online_fp32(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 7) // 8
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_online_fp32"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        for tile_index in pl.range(tile_begin, tile_end):
            row_start = tile_index * 8
            valid_rows = pl.min(8, rows - row_start)
            full_chunks = cols // 256
            tail_cols = cols % 256
            running_max = pl.slice(
                x, [8, 256], [row_start, 0],
                valid_shape=[valid_rows, 256],
            )

            running_max = pl.fillpad(running_max, pad_value=pl.PadValue.min)
            running_sum = pl.full([8, 256], dtype=pl.FP32, value=1.0)
            for chunk in pl.range(1, full_chunks):
                col_start = chunk * 256
                tile_x = pl.slice(
                    x, [8, 256], [row_start, col_start],
                    valid_shape=[valid_rows, 256],
                )
    
                tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
                new_max = pl.maximum(running_max, tile_x)
                correction = pl.exp(pl.sub(running_max, new_max))
                term = pl.exp(pl.sub(tile_x, new_max))
                running_sum = pl.add(pl.mul(running_sum, correction), term)
                running_max = new_max
            row_max = pl.row_max(running_max)
            correction = pl.exp(pl.row_expand_sub(running_max, row_max))
            denominator = pl.row_sum(pl.mul(running_sum, correction))
            denominator = pl.reshape(denominator, [1, 8])
            if tail_cols > 0:
                tail_x = pl.slice(
                    x, [8, 256], [row_start, full_chunks * 256],
                    valid_shape=[valid_rows, tail_cols],
                )
    
                tail_x = pl.fillpad(tail_x, pad_value=pl.PadValue.min)
                tail_max = pl.row_max(tail_x)
                tail_exp = pl.exp(pl.row_expand_sub(tail_x, tail_max))
                tail_sum = pl.reshape(pl.row_sum(tail_exp), [1, 8])
                merged_max = pl.maximum(row_max, tail_max)
                old_correction = pl.exp(pl.reshape(pl.sub(row_max, merged_max), [1, 8]))
                tail_correction = pl.exp(pl.reshape(pl.sub(tail_max, merged_max), [1, 8]))
                denominator = pl.add(
                    pl.mul(denominator, old_correction), pl.mul(tail_sum, tail_correction),
                )
                row_max = merged_max
            normalizer = pl.reshape(pl.recip(denominator), [8, 1])
            for chunk in pl.range((cols + 256 - 1) // 256):
                col_start = chunk * 256
                valid_cols = pl.min(256, cols - col_start)
                tile_x = pl.slice(
                    x, [8, 256], [row_start, col_start],
                    valid_shape=[valid_rows, valid_cols],
                )
    
                tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
                numerator = pl.exp(pl.row_expand_sub(tile_x, row_max))
                result = pl.row_expand_mul(numerator, normalizer)
    
                result = pl.set_validshape(result, valid_rows, valid_cols)
                y = pl.assemble(y, result, [row_start, col_start])
    return y


@pl.jit
def softmax_single_fp32(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    flat_y = pl.reshape(y, [1, rows])
    for offset in pl.parallel(0, rows, 128):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_single_fp32"):
            valid_cols = pl.min(128, rows - offset)
            result = pl.full([1, 128], dtype=pl.FP32, value=1.0)
            
            result = pl.set_validshape(result, 1, valid_cols)
            flat_y = pl.assemble(flat_y, result, [0, offset])
    return pl.reshape(flat_y, [rows, cols])


@pl.jit
def softmax_resident_bf16_1_s0(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_bf16_1_s0"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_bf16_1_s2(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_bf16_1_s2"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s0_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s0_w64_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s2_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s2_w64_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s0_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s0_w64_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s2_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s2_w64_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s0_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s0_w128_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s2_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s2_w128_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s0_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s0_w128_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s2_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s2_w128_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s0_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s0_w256_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s2_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s2_w256_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s0_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s0_w256_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_1_s2_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 0) // 1
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_1_s2_w256_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 1
            valid_rows = pl.min(1, rows - row_start)
            tile_x = pl.slice(
                x, [1, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 1, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_bf16_2_s0(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_bf16_2_s0"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_bf16_2_s2(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_bf16_2_s2"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s0_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s0_w64_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s2_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s2_w64_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s0_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s0_w64_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s2_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s2_w64_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s0_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s0_w128_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s2_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s2_w128_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s0_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s0_w128_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s2_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s2_w128_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s0_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s0_w256_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s2_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s2_w256_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s0_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s0_w256_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_2_s2_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 1) // 2
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_2_s2_w256_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 2
            valid_rows = pl.min(2, rows - row_start)
            tile_x = pl.slice(
                x, [2, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 2, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_bf16_4_s0(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_bf16_4_s0"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_resident_bf16_4_s2(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_resident_bf16_4_s2"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)
            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_x = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_x)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                lane_numerator = pl.reshape(numerator, [8, padded_cols // 8])
                lane_sum = pl.row_sum(lane_numerator)
                sum_summary = pl.reshape(lane_sum, [1, 8])
                sum_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 8)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s0_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s0_w64_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s2_w64_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s2_w64_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s0_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s0_w64_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s2_w64_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s2_w64_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 64, 64])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 64], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 64)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s0_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s0_w128_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s2_w128_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s2_w128_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s0_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s0_w128_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s2_w128_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s2_w128_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 128, 128])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 128], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 128)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s0_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s0_w256_k8"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s2_w256_k8(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s2_w256_k8"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [8, padded_cols // 8])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 8])
                max_carrier = pl.create_tensor([8, 8], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 8)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s0_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s0_w256_k16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.range(owner_tile_count):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_packet_bf16_4_s2_w256_k16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 3) // 4
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_packet_bf16_4_s2_w256_k16"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        owner_tile_count = tile_end - tile_begin
        for local_tile_index in pl.pipeline(owner_tile_count, stage=2):
            tile_index = tile_begin + local_tile_index
            row_start = tile_index * 4
            valid_rows = pl.min(4, rows - row_start)
            tile_x = pl.slice(
                x, [4, padded_cols], [row_start, 0],
                valid_shape=[valid_rows, cols],
            )
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_x = pl.set_validshape(tile_x, 4, padded_cols)

            for local_row in pl.range(valid_rows):
                row_x = pl.slice(tile_x, [1, padded_cols], [local_row, 0])
                lane_max_input = pl.reshape(row_x, [16, padded_cols // 16])
                lane_max = pl.row_max(lane_max_input)
                max_summary = pl.reshape(lane_max, [1, 16])
                max_carrier = pl.create_tensor([8, 16], dtype=pl.FP32)
                max_carrier = pl.assemble(max_carrier, max_summary, [0, 0])
                max_carrier = pl.set_validshape(max_carrier, 1, 16)
                max_final = pl.row_max(max_carrier)
                row_max = pl.read(max_final, [0, 0])
                numerator = pl.exp(pl.sub(row_x, row_max))
                sum_matrix = pl.reshape(numerator, [padded_cols // 256, 256])
                sum_summary = pl.col_sum(sum_matrix, is_binary=True)
                sum_carrier = pl.create_tensor([8, 256], dtype=pl.FP32)
                sum_carrier = pl.assemble(sum_carrier, sum_summary, [0, 0])
                sum_carrier = pl.set_validshape(sum_carrier, 1, 256)
                sum_final = pl.row_sum(sum_carrier)
                denominator = pl.read(sum_final, [0, 0])
                result = pl.div(numerator, denominator)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, 1, cols)
                y = pl.assemble(y, result, [row_start + local_row, 0])
    return y


@pl.jit
def softmax_online_bf16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    padded_cols = ((cols + 127) // 128) * 128
    tile_count = (rows + 7) // 8
    block_count = pl.min(96, tile_count)
    for block_index in pl.spmd(block_count, name_hint="softmax_online_bf16"):
        tile_begin = block_index * tile_count // block_count
        tile_end = (block_index + 1) * tile_count // block_count
        for tile_index in pl.range(tile_begin, tile_end):
            row_start = tile_index * 8
            valid_rows = pl.min(8, rows - row_start)
            full_chunks = cols // 256
            tail_cols = cols % 256
            running_max = pl.slice(
                x, [8, 256], [row_start, 0],
                valid_shape=[valid_rows, 256],
            )
            running_max = pl.cast(running_max, pl.FP32)
            running_max = pl.fillpad(running_max, pad_value=pl.PadValue.min)
            running_sum = pl.full([8, 256], dtype=pl.FP32, value=1.0)
            for chunk in pl.range(1, full_chunks):
                col_start = chunk * 256
                tile_x = pl.slice(
                    x, [8, 256], [row_start, col_start],
                    valid_shape=[valid_rows, 256],
                )
                tile_x = pl.cast(tile_x, pl.FP32)
                tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
                new_max = pl.maximum(running_max, tile_x)
                correction = pl.exp(pl.sub(running_max, new_max))
                term = pl.exp(pl.sub(tile_x, new_max))
                running_sum = pl.add(pl.mul(running_sum, correction), term)
                running_max = new_max
            row_max = pl.row_max(running_max)
            correction = pl.exp(pl.row_expand_sub(running_max, row_max))
            denominator = pl.row_sum(pl.mul(running_sum, correction))
            denominator = pl.reshape(denominator, [1, 8])
            if tail_cols > 0:
                tail_x = pl.slice(
                    x, [8, 256], [row_start, full_chunks * 256],
                    valid_shape=[valid_rows, tail_cols],
                )
                tail_x = pl.cast(tail_x, pl.FP32)
                tail_x = pl.fillpad(tail_x, pad_value=pl.PadValue.min)
                tail_max = pl.row_max(tail_x)
                tail_exp = pl.exp(pl.row_expand_sub(tail_x, tail_max))
                tail_sum = pl.reshape(pl.row_sum(tail_exp), [1, 8])
                merged_max = pl.maximum(row_max, tail_max)
                old_correction = pl.exp(pl.reshape(pl.sub(row_max, merged_max), [1, 8]))
                tail_correction = pl.exp(pl.reshape(pl.sub(tail_max, merged_max), [1, 8]))
                denominator = pl.add(
                    pl.mul(denominator, old_correction), pl.mul(tail_sum, tail_correction),
                )
                row_max = merged_max
            normalizer = pl.reshape(pl.recip(denominator), [8, 1])
            for chunk in pl.range((cols + 256 - 1) // 256):
                col_start = chunk * 256
                valid_cols = pl.min(256, cols - col_start)
                tile_x = pl.slice(
                    x, [8, 256], [row_start, col_start],
                    valid_shape=[valid_rows, valid_cols],
                )
                tile_x = pl.cast(tile_x, pl.FP32)
                tile_x = pl.fillpad(tile_x, pad_value=pl.PadValue.min)
                numerator = pl.exp(pl.row_expand_sub(tile_x, row_max))
                result = pl.row_expand_mul(numerator, normalizer)
                result = pl.cast(result, pl.BF16, mode="rint")
                result = pl.set_validshape(result, valid_rows, valid_cols)
                y = pl.assemble(y, result, [row_start, col_start])
    return y


@pl.jit
def softmax_single_bf16(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = x.shape
    flat_y = pl.reshape(y, [1, rows])
    for offset in pl.parallel(0, rows, 128):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_single_bf16"):
            valid_cols = pl.min(128, rows - offset)
            result = pl.full([1, 128], dtype=pl.FP32, value=1.0)
            result = pl.cast(result, pl.BF16, mode="rint")
            result = pl.set_validshape(result, 1, valid_cols)
            flat_y = pl.assemble(flat_y, result, [0, offset])
    return pl.reshape(flat_y, [rows, cols])



def _working_set_bytes(tile_rows, padded_cols, dtype, stage):
    storage_bytes = 2 if dtype == "bf16" else 4
    resident_fp32 = 4 * tile_rows * padded_cols
    staged_inputs = max(1, stage) * storage_bytes * tile_rows * padded_cols
    reduction_and_numerator = 8 * padded_cols
    output = storage_bytes * padded_cols
    return resident_fp32 + staged_inputs + reduction_and_numerator + output + 4096


def describe(shape):
    rows = int(shape["M"])
    cols = int(shape["N"])
    dtype = shape.get("dtype", "fp32")
    if rows < 1 or cols < 1:
        raise ValueError("Softmax requires positive M and N")
    if dtype not in ("fp32", "bf16"):
        raise ValueError("Softmax supports fp32 and bf16 storage")
    padded_cols = ((cols + 127) // 128) * 128
    tile_rows = REQUESTED_ROWS
    stage = REQUESTED_STAGE
    while tile_rows > 1 and (
        tile_rows * padded_cols > RESIDENT_ELEMENTS
        or _working_set_bytes(tile_rows, padded_cols, dtype, stage) > WORKING_SET_BUDGET_BYTES
    ):
        tile_rows //= 2
    if _working_set_bytes(tile_rows, padded_cols, dtype, stage) > WORKING_SET_BUDGET_BYTES:
        stage = 0
    if cols == 1:
        route = "singleton"
        tile_rows = 128
        stage = 0
        tile_count = (rows + 127) // 128
        task_count = tile_count
    elif padded_cols <= MAX_RESIDENT_COLUMNS and tile_rows * padded_cols <= RESIDENT_ELEMENTS:
        route = "resident" if stage == REQUESTED_STAGE else "resident_sequential_resource_fallback"
        tile_count = (rows + tile_rows - 1) // tile_rows
        task_count = min(PERSISTENT_BLOCKS, tile_count)
    else:
        route = "online_sequential"
        tile_rows = 8
        stage = 0
        tile_count = (rows + 7) // 8
        task_count = min(PERSISTENT_BLOCKS, tile_count)
    uses_aligned_epilogue = route.startswith("resident") and cols % 128 == 0
    effective_epilogue = EPILOGUE_MODE if uses_aligned_epilogue and not (
        EPILOGUE_MODE == 2 and dtype == "fp32"
    ) else 0
    effective_lane_width = 64
    while padded_cols // effective_lane_width > COLUMN_HEIGHT_LIMIT and effective_lane_width < REQUESTED_LANE_WIDTH:
        effective_lane_width *= 2
    while padded_cols % effective_lane_width:
        effective_lane_width //= 2
    effective_max_segments = REQUESTED_MAX_SEGMENTS
    while padded_cols % (8 * effective_max_segments):
        effective_max_segments //= 2
    if not uses_aligned_epilogue:
        effective_max_segments = 8
    return {
        "requested_max_segments": REQUESTED_MAX_SEGMENTS,
        "effective_max_segments": effective_max_segments,
        "max_batch": MAX_BATCH if uses_aligned_epilogue else 0,
        "max_final_carrier_shape": [8, effective_max_segments],
        "max_final_valid_shape": [tile_rows if MAX_BATCH and uses_aligned_epilogue else 1, effective_max_segments],
        "max_layout": MAX_LAYOUT if uses_aligned_epilogue else 0,
        "sum_layout": SUM_LAYOUT if uses_aligned_epilogue else 0,
        "requested_lane_width": REQUESTED_LANE_WIDTH,
        "column_height_limit": COLUMN_HEIGHT_LIMIT,
        "effective_column_height": padded_cols // effective_lane_width if uses_aligned_epilogue else None,
        "effective_lane_width": effective_lane_width if uses_aligned_epilogue else None,
        "max_first_reduction_shape": (
            [tile_rows * effective_max_segments if MAX_BATCH else effective_max_segments, padded_cols // effective_max_segments]
            if uses_aligned_epilogue else [8, padded_cols // 8]
        ),
        "sum_first_reduction_shape": (
            [padded_cols // effective_lane_width, effective_lane_width]
            if uses_aligned_epilogue and SUM_LAYOUT else [8, padded_cols // 8]
        ),
        "requested_epilogue_mode": EPILOGUE_MODE,
        "effective_epilogue_mode": effective_epilogue,
        "uses_aligned_epilogue": uses_aligned_epilogue,
        "aligned_input_min_padding": not uses_aligned_epilogue,
        "estimated_extra_output_packet_bytes": (
            (2 if dtype == "bf16" else 4) * tile_rows * padded_cols
            if effective_epilogue == 1 else 0
        ),
        "route": route,
        "dtype": dtype,
        "padded_cols": padded_cols,
        "requested_rows": REQUESTED_ROWS,
        "effective_rows": tile_rows,
        "requested_stage": REQUESTED_STAGE,
        "effective_stage": stage,
        "tile_count": tile_count,
        "task_count": task_count,
        "ownership_mode": OWNERSHIP_MODE,
        "requested_input_cache_policy": "BYPASS" if INPUT_BYPASS else "DEFAULT",
        "effective_input_cache_policy": (
            "BYPASS" if INPUT_BYPASS and route == "resident" else "DEFAULT"
        ),
        "output_cache_policy": "DEFAULT",
        "owner_tiles_floor": tile_count // task_count,
        "owner_tiles_ceil": (tile_count + task_count - 1) // task_count,
        "owners_with_extra_tile": tile_count % task_count,
        "pipeline_remainder_owners": sum(
            ((tile_count // task_count) + (owner < tile_count % task_count)) % stage != 0
            for owner in range(task_count)
        ) if stage else 0,
        "reduction_mode": REDUCTION_MODE,
        "summary_physical_shape": [8, 8] if REDUCTION_MODE else None,
        "summary_valid_shape": [1, 8] if REDUCTION_MODE else None,
        "hierarchical_statistics": ("max", "sum") if REDUCTION_MODE == 2 else ("max",) if REDUCTION_MODE == 1 else (),
        "resource_model_status": "Inherited source estimate, not an allocator upper bound; inspect hierarchy scratch and aliases.",
        "estimated_resident_working_set_bytes": (
            _working_set_bytes(tile_rows, padded_cols, dtype, stage) if route.startswith("resident") else None
        ),
        "working_set_budget_bytes": WORKING_SET_BUDGET_BYTES,
    }


def golden_softmax(tensors):
    import torch

    tensors["y"][:] = torch.softmax(tensors["x"].float(), dim=-1).to(tensors["y"].dtype)


def build(shape):
    import torch
    from golden import TensorSpec

    route = describe(shape)
    rows = int(shape["M"])
    cols = int(shape["N"])
    dtype = route["dtype"]
    storage = torch.bfloat16 if dtype == "bf16" else torch.float32
    specs = [
        TensorSpec("x", [rows, cols], storage, init_value=torch.randn),
        TensorSpec("y", [rows, cols], storage),
    ]
    resident_kernels = {
        ("fp32", 1, 0): softmax_resident_fp32_1_s0,
        ("fp32", 1, 2): softmax_resident_fp32_1_s2,
        ("fp32", 2, 0): softmax_resident_fp32_2_s0,
        ("fp32", 2, 2): softmax_resident_fp32_2_s2,
        ("fp32", 4, 0): softmax_resident_fp32_4_s0,
        ("fp32", 4, 2): softmax_resident_fp32_4_s2,
        ("bf16", 1, 0): softmax_resident_bf16_1_s0,
        ("bf16", 1, 2): softmax_resident_bf16_1_s2,
        ("bf16", 2, 0): softmax_resident_bf16_2_s0,
        ("bf16", 2, 2): softmax_resident_bf16_2_s2,
        ("bf16", 4, 0): softmax_resident_bf16_4_s0,
        ("bf16", 4, 2): softmax_resident_bf16_4_s2,
    }
    packet_kernels = {
        ("fp32", 1, 0, 64, 8): softmax_packet_fp32_1_s0_w64_k8,
        ("fp32", 1, 0, 64, 16): softmax_packet_fp32_1_s0_w64_k16,
        ("fp32", 1, 0, 128, 8): softmax_packet_fp32_1_s0_w128_k8,
        ("fp32", 1, 0, 128, 16): softmax_packet_fp32_1_s0_w128_k16,
        ("fp32", 1, 0, 256, 8): softmax_packet_fp32_1_s0_w256_k8,
        ("fp32", 1, 0, 256, 16): softmax_packet_fp32_1_s0_w256_k16,
        ("fp32", 1, 2, 64, 8): softmax_packet_fp32_1_s2_w64_k8,
        ("fp32", 1, 2, 64, 16): softmax_packet_fp32_1_s2_w64_k16,
        ("fp32", 1, 2, 128, 8): softmax_packet_fp32_1_s2_w128_k8,
        ("fp32", 1, 2, 128, 16): softmax_packet_fp32_1_s2_w128_k16,
        ("fp32", 1, 2, 256, 8): softmax_packet_fp32_1_s2_w256_k8,
        ("fp32", 1, 2, 256, 16): softmax_packet_fp32_1_s2_w256_k16,
        ("fp32", 2, 0, 64, 8): softmax_packet_fp32_2_s0_w64_k8,
        ("fp32", 2, 0, 64, 16): softmax_packet_fp32_2_s0_w64_k16,
        ("fp32", 2, 0, 128, 8): softmax_packet_fp32_2_s0_w128_k8,
        ("fp32", 2, 0, 128, 16): softmax_packet_fp32_2_s0_w128_k16,
        ("fp32", 2, 0, 256, 8): softmax_packet_fp32_2_s0_w256_k8,
        ("fp32", 2, 0, 256, 16): softmax_packet_fp32_2_s0_w256_k16,
        ("fp32", 2, 2, 64, 8): softmax_packet_fp32_2_s2_w64_k8,
        ("fp32", 2, 2, 64, 16): softmax_packet_fp32_2_s2_w64_k16,
        ("fp32", 2, 2, 128, 8): softmax_packet_fp32_2_s2_w128_k8,
        ("fp32", 2, 2, 128, 16): softmax_packet_fp32_2_s2_w128_k16,
        ("fp32", 2, 2, 256, 8): softmax_packet_fp32_2_s2_w256_k8,
        ("fp32", 2, 2, 256, 16): softmax_packet_fp32_2_s2_w256_k16,
        ("fp32", 4, 0, 64, 8): softmax_packet_fp32_4_s0_w64_k8,
        ("fp32", 4, 0, 64, 16): softmax_packet_fp32_4_s0_w64_k16,
        ("fp32", 4, 0, 128, 8): softmax_packet_fp32_4_s0_w128_k8,
        ("fp32", 4, 0, 128, 16): softmax_packet_fp32_4_s0_w128_k16,
        ("fp32", 4, 0, 256, 8): softmax_packet_fp32_4_s0_w256_k8,
        ("fp32", 4, 0, 256, 16): softmax_packet_fp32_4_s0_w256_k16,
        ("fp32", 4, 2, 64, 8): softmax_packet_fp32_4_s2_w64_k8,
        ("fp32", 4, 2, 64, 16): softmax_packet_fp32_4_s2_w64_k16,
        ("fp32", 4, 2, 128, 8): softmax_packet_fp32_4_s2_w128_k8,
        ("fp32", 4, 2, 128, 16): softmax_packet_fp32_4_s2_w128_k16,
        ("fp32", 4, 2, 256, 8): softmax_packet_fp32_4_s2_w256_k8,
        ("fp32", 4, 2, 256, 16): softmax_packet_fp32_4_s2_w256_k16,
        ("bf16", 1, 0, 64, 8): softmax_packet_bf16_1_s0_w64_k8,
        ("bf16", 1, 0, 64, 16): softmax_packet_bf16_1_s0_w64_k16,
        ("bf16", 1, 0, 128, 8): softmax_packet_bf16_1_s0_w128_k8,
        ("bf16", 1, 0, 128, 16): softmax_packet_bf16_1_s0_w128_k16,
        ("bf16", 1, 0, 256, 8): softmax_packet_bf16_1_s0_w256_k8,
        ("bf16", 1, 0, 256, 16): softmax_packet_bf16_1_s0_w256_k16,
        ("bf16", 1, 2, 64, 8): softmax_packet_bf16_1_s2_w64_k8,
        ("bf16", 1, 2, 64, 16): softmax_packet_bf16_1_s2_w64_k16,
        ("bf16", 1, 2, 128, 8): softmax_packet_bf16_1_s2_w128_k8,
        ("bf16", 1, 2, 128, 16): softmax_packet_bf16_1_s2_w128_k16,
        ("bf16", 1, 2, 256, 8): softmax_packet_bf16_1_s2_w256_k8,
        ("bf16", 1, 2, 256, 16): softmax_packet_bf16_1_s2_w256_k16,
        ("bf16", 2, 0, 64, 8): softmax_packet_bf16_2_s0_w64_k8,
        ("bf16", 2, 0, 64, 16): softmax_packet_bf16_2_s0_w64_k16,
        ("bf16", 2, 0, 128, 8): softmax_packet_bf16_2_s0_w128_k8,
        ("bf16", 2, 0, 128, 16): softmax_packet_bf16_2_s0_w128_k16,
        ("bf16", 2, 0, 256, 8): softmax_packet_bf16_2_s0_w256_k8,
        ("bf16", 2, 0, 256, 16): softmax_packet_bf16_2_s0_w256_k16,
        ("bf16", 2, 2, 64, 8): softmax_packet_bf16_2_s2_w64_k8,
        ("bf16", 2, 2, 64, 16): softmax_packet_bf16_2_s2_w64_k16,
        ("bf16", 2, 2, 128, 8): softmax_packet_bf16_2_s2_w128_k8,
        ("bf16", 2, 2, 128, 16): softmax_packet_bf16_2_s2_w128_k16,
        ("bf16", 2, 2, 256, 8): softmax_packet_bf16_2_s2_w256_k8,
        ("bf16", 2, 2, 256, 16): softmax_packet_bf16_2_s2_w256_k16,
        ("bf16", 4, 0, 64, 8): softmax_packet_bf16_4_s0_w64_k8,
        ("bf16", 4, 0, 64, 16): softmax_packet_bf16_4_s0_w64_k16,
        ("bf16", 4, 0, 128, 8): softmax_packet_bf16_4_s0_w128_k8,
        ("bf16", 4, 0, 128, 16): softmax_packet_bf16_4_s0_w128_k16,
        ("bf16", 4, 0, 256, 8): softmax_packet_bf16_4_s0_w256_k8,
        ("bf16", 4, 0, 256, 16): softmax_packet_bf16_4_s0_w256_k16,
        ("bf16", 4, 2, 64, 8): softmax_packet_bf16_4_s2_w64_k8,
        ("bf16", 4, 2, 64, 16): softmax_packet_bf16_4_s2_w64_k16,
        ("bf16", 4, 2, 128, 8): softmax_packet_bf16_4_s2_w128_k8,
        ("bf16", 4, 2, 128, 16): softmax_packet_bf16_4_s2_w128_k16,
        ("bf16", 4, 2, 256, 8): softmax_packet_bf16_4_s2_w256_k8,
        ("bf16", 4, 2, 256, 16): softmax_packet_bf16_4_s2_w256_k16,
    }
    single_kernels = {"fp32": softmax_single_fp32, "bf16": softmax_single_bf16}
    online_kernels = {"fp32": softmax_online_fp32, "bf16": softmax_online_bf16}
    if route["route"] == "singleton":
        kernel = single_kernels[dtype]
    elif route["route"].startswith("resident"):
        if route["uses_aligned_epilogue"]:
            kernel = packet_kernels[dtype, route["effective_rows"], route["effective_stage"], route["effective_lane_width"], route["effective_max_segments"]]
        else:
            kernel = resident_kernels[dtype, route["effective_rows"], route["effective_stage"]]
    else:
        kernel = online_kernels[dtype]
    rtol, atol = (1e-2, 1e-6) if dtype == "bf16" else (1e-4, 1e-4)
    return kernel, specs, golden_softmax, rtol, atol

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
