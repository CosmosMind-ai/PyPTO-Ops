# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm forward — storage-dtype I/O with FP32 reduction."""
import pypto.language as pl


NARROW_CUTOFF = 3072
MIDDLE_MAX = 4096
WIDE_POLICY = 1
ULTRA_RESIDENT = 1
ULTRA_MIN = 14336
NARROW_COL_TILE = 512
NARROW_DISPATCH_BLOCKS = 48
NARROW_ENGINE = 1
NARROW_ROW_TILE = 16
RETENTION_MODE = 1
GROUP_PIPELINE = 0
ROW_TILE = 8
COL_TILE = 512
REDUCTION_MODE = 1
OUTPUT_PIPELINE = 1
DISPATCH_BLOCKS = 48
VEC_LIMIT_BYTES = 184 * 1024


def _fallback_geometry(shape):
    rows = int(shape["M"])
    hidden = int(shape["N"])
    if rows <= 0 or hidden <= 0:
        raise ValueError("M and N must be positive")
    storage_bytes = {"bf16": 2, "fp32": 4}[shape["dtype"]]
    row_tile = ROW_TILE
    narrow_band = shape["dtype"] == "bf16" and 1 < hidden < NARROW_CUTOFF
    requested_col_tile = NARROW_COL_TILE if narrow_band else COL_TILE
    requested_row_tile = NARROW_ROW_TILE if narrow_band and hidden > 1 else ROW_TILE
    if requested_row_tile > ROW_TILE:
        proposed_width = (hidden + requested_col_tile - 1) // requested_col_tile * requested_col_tile
        proposed_cache = requested_row_tile * proposed_width * storage_bytes
        proposed_statistics = requested_row_tile * requested_col_tile * (storage_bytes + 12)
        proposed_normalization = requested_row_tile * requested_col_tile * (4 + 2 * storage_bytes) * (1 + OUTPUT_PIPELINE)
        proposed_bytes = proposed_cache + max(proposed_statistics, proposed_normalization) + 16 * requested_row_tile + 4 * 1024
        if proposed_bytes <= VEC_LIMIT_BYTES:
            row_tile = requested_row_tile
    col_tile = requested_col_tile
    while col_tile > COL_TILE:
        proposed_width = (hidden + col_tile - 1) // col_tile * col_tile
        proposed_cache = row_tile * proposed_width * storage_bytes
        proposed_statistics = row_tile * col_tile * (storage_bytes + 12)
        proposed_normalization = row_tile * col_tile * (4 + 2 * storage_bytes) * (1 + OUTPUT_PIPELINE)
        proposed_bytes = proposed_cache + max(proposed_statistics, proposed_normalization) + 16 * row_tile + 4 * 1024
        if proposed_bytes <= VEC_LIMIT_BYTES:
            break
        col_tile //= 2
    cached_width = (hidden + col_tile - 1) // col_tile * col_tile
    chunk_elements = row_tile * col_tile
    cache_bytes = row_tile * cached_width * storage_bytes
    statistic_bytes = 16 * row_tile
    reserve_bytes = 4 * 1024
    statistics_element_bytes = storage_bytes + 12 if REDUCTION_MODE else max(storage_bytes + 4, 8)
    statistics_bytes = chunk_elements * statistics_element_bytes
    normalization_bytes = chunk_elements * (4 + 2 * storage_bytes) * (1 + OUTPUT_PIPELINE)
    compact_bytes = cache_bytes + max(statistics_bytes, normalization_bytes) + statistic_bytes + reserve_bytes
    streaming_bytes = chunk_elements * (2 * storage_bytes + 12) + statistic_bytes + 16 * 1024
    mode = "compact" if compact_bytes <= VEC_LIMIT_BYTES else "streaming"
    if mode == "streaming":
        while col_tile > 128 and streaming_bytes > VEC_LIMIT_BYTES:
            col_tile //= 2
            chunk_elements = row_tile * col_tile
            streaming_bytes = chunk_elements * (2 * storage_bytes + 12) + statistic_bytes + 16 * 1024
        cached_width = (hidden + col_tile - 1) // col_tile * col_tile
    if hidden == 1:
        mode = "scalar" if rows == 1 else "unit"
        row_tile = 16
        cached_width = 1
    row_groups = (rows + row_tile - 1) // row_tile
    dispatch_blocks = NARROW_DISPATCH_BLOCKS if narrow_band else DISPATCH_BLOCKS
    blocks = min(dispatch_blocks, row_groups)
    return dict(rows=rows, hidden=hidden, row_tile=row_tile, col_tile=col_tile,
                cached_width=cached_width, blocks=blocks, mode=mode,
                requested_row_tile=requested_row_tile,
                selected_compact_row_tile=row_tile if mode == "compact" else None,
                requested_col_tile=requested_col_tile, requested_reduction_mode=REDUCTION_MODE,
                selected_reduction_mode=REDUCTION_MODE if mode == "compact" else 0,
                requested_output_pipeline=OUTPUT_PIPELINE,
                selected_output_pipeline=OUTPUT_PIPELINE if mode == "compact" else 0,
                cache_bytes=cache_bytes, statistics_bytes=statistics_bytes,
                normalization_bytes=normalization_bytes, compact_bytes=compact_bytes,
                streaming_bytes=streaming_bytes, vec_limit_bytes=VEC_LIMIT_BYTES)


def _fullwidth_bound(logical_rows, padded_hidden, phase_aware):
    elements = logical_rows * padded_hidden
    cache_bytes = 2 * elements
    fp32_work_bytes = 4 * elements
    reduction_scratch_bytes = 4 * elements
    output_bytes = 2 * elements
    statistics_phase_bytes = cache_bytes + fp32_work_bytes + reduction_scratch_bytes
    normalization_phase_bytes = cache_bytes + fp32_work_bytes + output_bytes
    phase_bytes = max(statistics_phase_bytes, normalization_phase_bytes)
    return (phase_bytes if phase_aware else 12 * elements) + 8 * 1024


def _donor_geometry(shape):
    layout = _fallback_geometry(shape)
    hidden = layout["hidden"]
    middle_band = NARROW_CUTOFF <= hidden <= MIDDLE_MAX
    ultra_band = hidden >= ULTRA_MIN
    requested_rows = 4 if middle_band else (1 if ultra_band else 2)
    layout.update(requested_logical_rows=requested_rows, retention_mode=RETENTION_MODE,
                  wide_policy=WIDE_POLICY, ultra_resident=ULTRA_RESIDENT,
                  narrow_cutoff=NARROW_CUTOFF, middle_max=MIDDLE_MAX,
                  requested_group_pipeline=0, selected_group_pipeline=0,
                  route_reason="inherited", residency_attempts=[])
    if shape["dtype"] == "bf16" and 1 < hidden < NARROW_CUTOFF:
        layout.update(narrow_engine=NARROW_ENGINE, native_attempts=[])
        if NARROW_ENGINE:
            padded_hidden = (hidden + 15) // 16 * 16
            choices = (16, 8) if NARROW_ROW_TILE == 16 else (8,)
            for native_rows in choices:
                native_bytes = _fullwidth_bound(native_rows, padded_hidden, True)
                layout["native_attempts"].append(dict(
                    row_tile=native_rows, padded_hidden=padded_hidden,
                    fullwidth_bytes=native_bytes, fits=native_bytes <= VEC_LIMIT_BYTES))
                if native_bytes <= VEC_LIMIT_BYTES:
                    layout.update(mode="native_narrow", row_tile=native_rows,
                                  padded_hidden=padded_hidden, cached_width=padded_hidden,
                                  col_tile=padded_hidden, fullwidth_bytes=native_bytes,
                                  blocks=min(DISPATCH_BLOCKS, (layout["rows"] + native_rows - 1) // native_rows),
                                  selected_output_pipeline=0, selected_reduction_mode=-1,
                                  route_reason="native_narrow")
                    return layout
            layout["route_reason"] = "native_capacity_compact_fallback"
        else:
            layout["route_reason"] = "narrow_compact_control"
        return layout
    if shape["dtype"] != "bf16" or hidden < NARROW_CUTOFF:
        return layout
    if ultra_band and not ULTRA_RESIDENT:
        layout["route_reason"] = "ultra_inherited"
        return layout
    choices = (4, 2) if middle_band else ((1,) if ultra_band else (2, 1))
    for logical_rows in choices:
        segments = 8 // logical_rows
        alignment = 16 * segments
        padded_hidden = (hidden + alignment - 1) // alignment * alignment
        phase_aware = bool(middle_band or ultra_band or WIDE_POLICY == 2)
        fullwidth_bytes = _fullwidth_bound(logical_rows, padded_hidden, phase_aware)
        layout["residency_attempts"].append(dict(
            logical_rows=logical_rows, padded_hidden=padded_hidden,
            phase_aware=int(phase_aware), fullwidth_bytes=fullwidth_bytes,
            fits=fullwidth_bytes <= VEC_LIMIT_BYTES))
        if fullwidth_bytes <= VEC_LIMIT_BYTES:
            layout.update(mode="segmented", row_tile=logical_rows,
                          cached_width=padded_hidden, padded_hidden=padded_hidden,
                          segment_width=padded_hidden // segments, segments=segments,
                          blocks=min(DISPATCH_BLOCKS, (layout["rows"] + logical_rows - 1) // logical_rows),
                          fullwidth_bytes=fullwidth_bytes, selected_output_pipeline=0,
                          selected_reduction_mode=-1, selected_phase_aware=int(phase_aware),
                          route_reason="middle" if middle_band else ("ultra" if ultra_band else "wide"))
            return layout
    layout.update(fullwidth_bytes=fullwidth_bytes, route_reason="residency_fallback")
    return layout


def _make_compact_bf16(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Compact_Bf16RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="compact_bf16_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    cache_valid = pl.load(x, [row_start, 0], [row_tile, cached_width], valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec)
                    source_cache = pl.tile.fillpad_inplace(cache_valid, pad_value=pl.PadValue.zero)
                    lane_sum = pl.tile.full([row_tile, col_tile], dtype=pl.FP32, value=0.0)
                    for col_start in pl.range(0, cached_width, col_tile):
                        source_chunk = pl.tile.slice(source_cache, [row_tile, col_tile], [0, col_start])
                        source_fp32 = pl.tile.cast(source_chunk, target_type=pl.FP32)
                        squared = pl.tile.mul(source_fp32, source_fp32)
                        lane_sum = pl.tile.add(lane_sum, squared)
                    reduction_scratch = pl.tile.create([row_tile, col_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    square_column = pl.tile.row_sum(lane_sum, reduction_scratch)
                    square_sum = pl.tile.reshape(square_column, [1, row_tile])
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [row_tile, 1])
                    for col_start in pl.pipeline(0, cached_width, col_tile, stage=2):
                        output_chunk = pl.tile.slice(source_cache, [row_tile, col_tile], [0, col_start])
                        output_fp32 = pl.tile.cast(output_chunk, target_type=pl.FP32)
                        normalized = pl.tile.row_expand_mul(output_fp32, inverse_column)
                        stored = pl.tile.cast(normalized, target_type=storage_type)
                        valid_cols = pl.min(col_tile, hidden - col_start)
                        stored_valid = pl.tile.set_validshape(stored, valid_rows, valid_cols)
                        y = pl.store(stored_valid, [row_start, col_start], y)
            return y

    return Compact_Bf16RMSNorm


def _make_streaming_bf16(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Streaming_Bf16RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="streaming_bf16_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    square_sum = pl.tile.full([1, row_tile], dtype=pl.FP32, value=0.0)
                    for col_start in pl.range(0, hidden, col_tile):
                        valid_cols = pl.min(col_tile, hidden - col_start)
                        source_valid = pl.load(x, [row_start, col_start], [row_tile, col_tile], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec)
                        source_chunk = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                        source_fp32 = pl.tile.cast(source_chunk, target_type=pl.FP32)
                        squared = pl.tile.mul(source_fp32, source_fp32)
                        reduction_scratch = pl.tile.create([row_tile, col_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        partial_column = pl.tile.row_sum(squared, reduction_scratch)
                        partial_row = pl.tile.reshape(partial_column, [1, row_tile])
                        square_sum = pl.tile.add(square_sum, partial_row)
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [row_tile, 1])
                    for col_start in pl.range(0, hidden, col_tile):
                        valid_cols = pl.min(col_tile, hidden - col_start)
                        output_chunk = pl.load(x, [row_start, col_start], [row_tile, col_tile], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec)
                        output_fp32 = pl.tile.cast(output_chunk, target_type=pl.FP32)
                        normalized = pl.tile.row_expand_mul(output_fp32, inverse_column)
                        stored = pl.tile.cast(normalized, target_type=storage_type)
                        stored_valid = pl.tile.set_validshape(stored, valid_rows, valid_cols)
                        y = pl.store(stored_valid, [row_start, col_start], y)
            return y

    return Streaming_Bf16RMSNorm


def _make_unit_bf16(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Unit_Bf16RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            flat_x = pl.reshape(x, [1, rows])
            flat_y = pl.reshape(y, [1, rows])
            for block in pl.spmd(blocks, name_hint="unit_bf16_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    source_valid = pl.load(flat_x, [0, row_start], [1, row_tile], valid_shape=[1, valid_rows], target_memory=pl.MemorySpace.Vec)
                    source_chunk = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                    source_fp32 = pl.tile.cast(source_chunk, target_type=pl.FP32)
                    squared = pl.tile.mul(source_fp32, source_fp32)
                    stabilized = pl.tile.adds(squared, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    normalized = pl.tile.mul(source_fp32, inverse_row)
                    stored = pl.tile.cast(normalized, target_type=storage_type)
                    stored_valid = pl.tile.set_validshape(stored, 1, valid_rows)
                    flat_y = pl.store(stored_valid, [0, row_start], flat_y)
            return pl.reshape(flat_y, [rows, hidden])

    return Unit_Bf16RMSNorm


def _make_scalar_bf16(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Scalar_Bf16RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="scalar_bf16_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    input_value = pl.read(x, [0, 0])
                    source_storage = pl.tile.full([1, row_tile], dtype=storage_type, value=0.0)
                    pl.tile.write(source_storage, [0, 0], input_value)
                    source_fp32 = pl.tile.cast(source_storage, target_type=pl.FP32)
                    squared = pl.tile.mul(source_fp32, source_fp32)
                    stabilized = pl.tile.adds(squared, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    normalized = pl.tile.mul(source_fp32, inverse_row)
                    stored = pl.tile.cast(normalized, target_type=storage_type)
                    output_value = pl.tile.read(stored, [0, 0])
                    pl.write(y, [0, 0], output_value)
            return y

    return Scalar_Bf16RMSNorm


def _make_compact_fp32(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Compact_Fp32RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="compact_fp32_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    cache_valid = pl.load(x, [row_start, 0], [row_tile, cached_width], valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec)
                    source_cache = pl.tile.fillpad_inplace(cache_valid, pad_value=pl.PadValue.zero)
                    lane_sum = pl.tile.full([row_tile, col_tile], dtype=pl.FP32, value=0.0)
                    for col_start in pl.range(0, cached_width, col_tile):
                        source_chunk = pl.tile.slice(source_cache, [row_tile, col_tile], [0, col_start])
                        source_fp32 = source_chunk
                        squared = pl.tile.mul(source_fp32, source_fp32)
                        lane_sum = pl.tile.add(lane_sum, squared)
                    reduction_scratch = pl.tile.create([row_tile, col_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    square_column = pl.tile.row_sum(lane_sum, reduction_scratch)
                    square_sum = pl.tile.reshape(square_column, [1, row_tile])
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [row_tile, 1])
                    for col_start in pl.pipeline(0, cached_width, col_tile, stage=2):
                        output_chunk = pl.tile.slice(source_cache, [row_tile, col_tile], [0, col_start])
                        output_fp32 = output_chunk
                        normalized = pl.tile.row_expand_mul(output_fp32, inverse_column)
                        stored = normalized
                        valid_cols = pl.min(col_tile, hidden - col_start)
                        stored_valid = pl.tile.set_validshape(stored, valid_rows, valid_cols)
                        y = pl.store(stored_valid, [row_start, col_start], y)
            return y

    return Compact_Fp32RMSNorm


def _make_streaming_fp32(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Streaming_Fp32RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="streaming_fp32_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    square_sum = pl.tile.full([1, row_tile], dtype=pl.FP32, value=0.0)
                    for col_start in pl.range(0, hidden, col_tile):
                        valid_cols = pl.min(col_tile, hidden - col_start)
                        source_valid = pl.load(x, [row_start, col_start], [row_tile, col_tile], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec)
                        source_chunk = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                        source_fp32 = source_chunk
                        squared = pl.tile.mul(source_fp32, source_fp32)
                        reduction_scratch = pl.tile.create([row_tile, col_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        partial_column = pl.tile.row_sum(squared, reduction_scratch)
                        partial_row = pl.tile.reshape(partial_column, [1, row_tile])
                        square_sum = pl.tile.add(square_sum, partial_row)
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [row_tile, 1])
                    for col_start in pl.range(0, hidden, col_tile):
                        valid_cols = pl.min(col_tile, hidden - col_start)
                        output_chunk = pl.load(x, [row_start, col_start], [row_tile, col_tile], valid_shape=[valid_rows, valid_cols], target_memory=pl.MemorySpace.Vec)
                        output_fp32 = output_chunk
                        normalized = pl.tile.row_expand_mul(output_fp32, inverse_column)
                        stored = normalized
                        stored_valid = pl.tile.set_validshape(stored, valid_rows, valid_cols)
                        y = pl.store(stored_valid, [row_start, col_start], y)
            return y

    return Streaming_Fp32RMSNorm


def _make_unit_fp32(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Unit_Fp32RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            flat_x = pl.reshape(x, [1, rows])
            flat_y = pl.reshape(y, [1, rows])
            for block in pl.spmd(blocks, name_hint="unit_fp32_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    source_valid = pl.load(flat_x, [0, row_start], [1, row_tile], valid_shape=[1, valid_rows], target_memory=pl.MemorySpace.Vec)
                    source_chunk = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                    source_fp32 = source_chunk
                    squared = pl.tile.mul(source_fp32, source_fp32)
                    stabilized = pl.tile.adds(squared, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    normalized = pl.tile.mul(source_fp32, inverse_row)
                    stored = normalized
                    stored_valid = pl.tile.set_validshape(stored, 1, valid_rows)
                    flat_y = pl.store(stored_valid, [0, row_start], flat_y)
            return pl.reshape(flat_y, [rows, hidden])

    return Unit_Fp32RMSNorm


def _make_scalar_fp32(rows, hidden, row_tile, col_tile, cached_width, blocks, storage_type):
    row_groups = (rows + row_tile - 1) // row_tile

    @pl.program
    class Scalar_Fp32RMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="scalar_fp32_rows"):
                for group in pl.range(block, row_groups, blocks):
                    row_start = group * row_tile
                    input_value = pl.read(x, [0, 0])
                    source_storage = pl.tile.full([1, row_tile], dtype=storage_type, value=0.0)
                    pl.tile.write(source_storage, [0, 0], input_value)
                    source_fp32 = source_storage
                    squared = pl.tile.mul(source_fp32, source_fp32)
                    stabilized = pl.tile.adds(squared, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    normalized = pl.tile.mul(source_fp32, inverse_row)
                    stored = normalized
                    output_value = pl.tile.read(stored, [0, 0])
                    pl.write(y, [0, 0], output_value)
            return y

    return Scalar_Fp32RMSNorm



def _make_segmented_1(rows, hidden, row_tile, padded_hidden, blocks, storage_type):
    segment_width = padded_hidden // 8

    @pl.program
    class SegmentedRMSNorm1:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="segment_rows"):
                indices_1 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_1, [0, 0], pl.const(1, pl.INT32))
                pl.tile.write(indices_1, [0, 1], pl.const(0, pl.INT32))
                pl.tile.write(indices_1, [0, 2], pl.const(3, pl.INT32))
                pl.tile.write(indices_1, [0, 3], pl.const(2, pl.INT32))
                pl.tile.write(indices_1, [0, 4], pl.const(5, pl.INT32))
                pl.tile.write(indices_1, [0, 5], pl.const(4, pl.INT32))
                pl.tile.write(indices_1, [0, 6], pl.const(7, pl.INT32))
                pl.tile.write(indices_1, [0, 7], pl.const(6, pl.INT32))
                indices_2 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_2, [0, 0], pl.const(2, pl.INT32))
                pl.tile.write(indices_2, [0, 1], pl.const(3, pl.INT32))
                pl.tile.write(indices_2, [0, 2], pl.const(0, pl.INT32))
                pl.tile.write(indices_2, [0, 3], pl.const(1, pl.INT32))
                pl.tile.write(indices_2, [0, 4], pl.const(6, pl.INT32))
                pl.tile.write(indices_2, [0, 5], pl.const(7, pl.INT32))
                pl.tile.write(indices_2, [0, 6], pl.const(4, pl.INT32))
                pl.tile.write(indices_2, [0, 7], pl.const(5, pl.INT32))
                indices_4 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_4, [0, 0], pl.const(4, pl.INT32))
                pl.tile.write(indices_4, [0, 1], pl.const(5, pl.INT32))
                pl.tile.write(indices_4, [0, 2], pl.const(6, pl.INT32))
                pl.tile.write(indices_4, [0, 3], pl.const(7, pl.INT32))
                pl.tile.write(indices_4, [0, 4], pl.const(0, pl.INT32))
                pl.tile.write(indices_4, [0, 5], pl.const(1, pl.INT32))
                pl.tile.write(indices_4, [0, 6], pl.const(2, pl.INT32))
                pl.tile.write(indices_4, [0, 7], pl.const(3, pl.INT32))
                gather_scratch = pl.tile.create([1, 8], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                for group in pl.range(block, (rows + row_tile - 1) // row_tile, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    source_valid = pl.load(x, [row_start, 0], [row_tile, padded_hidden], valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec)
                    source_storage = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                    source_fp32 = pl.tile.cast(source_storage, target_type=pl.FP32)
                    source_segments = pl.tile.reshape(source_fp32, [8, segment_width])
                    squared = pl.tile.mul(source_segments, source_segments)
                    reduction_scratch = pl.tile.create([8, segment_width], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    partial_column = pl.tile.row_sum(squared, reduction_scratch)
                    square_sum = pl.tile.reshape(partial_column, [1, 8])
                    gathered_1 = pl.tile.gather(square_sum, indices_1, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_1)
                    gathered_2 = pl.tile.gather(square_sum, indices_2, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_2)
                    gathered_4 = pl.tile.gather(square_sum, indices_4, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_4)
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [8, 1])
                    output_source = pl.tile.reshape(pl.tile.cast(source_storage, target_type=pl.FP32), [8, segment_width])
                    normalized = pl.tile.row_expand_mul(output_source, inverse_column)
                    logical_output = pl.tile.reshape(normalized, [row_tile, padded_hidden])
                    stored = pl.tile.cast(logical_output, target_type=storage_type)
                    stored_valid = pl.tile.set_validshape(stored, valid_rows, hidden)
                    y = pl.store(stored_valid, [row_start, 0], y)
            return y

    return SegmentedRMSNorm1


def _make_segmented_2(rows, hidden, row_tile, padded_hidden, blocks, storage_type):
    segment_width = padded_hidden // 4

    @pl.program
    class SegmentedRMSNorm2:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="segment_rows"):
                indices_1 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_1, [0, 0], pl.const(1, pl.INT32))
                pl.tile.write(indices_1, [0, 1], pl.const(0, pl.INT32))
                pl.tile.write(indices_1, [0, 2], pl.const(3, pl.INT32))
                pl.tile.write(indices_1, [0, 3], pl.const(2, pl.INT32))
                pl.tile.write(indices_1, [0, 4], pl.const(5, pl.INT32))
                pl.tile.write(indices_1, [0, 5], pl.const(4, pl.INT32))
                pl.tile.write(indices_1, [0, 6], pl.const(7, pl.INT32))
                pl.tile.write(indices_1, [0, 7], pl.const(6, pl.INT32))
                indices_2 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_2, [0, 0], pl.const(2, pl.INT32))
                pl.tile.write(indices_2, [0, 1], pl.const(3, pl.INT32))
                pl.tile.write(indices_2, [0, 2], pl.const(0, pl.INT32))
                pl.tile.write(indices_2, [0, 3], pl.const(1, pl.INT32))
                pl.tile.write(indices_2, [0, 4], pl.const(6, pl.INT32))
                pl.tile.write(indices_2, [0, 5], pl.const(7, pl.INT32))
                pl.tile.write(indices_2, [0, 6], pl.const(4, pl.INT32))
                pl.tile.write(indices_2, [0, 7], pl.const(5, pl.INT32))
                gather_scratch = pl.tile.create([1, 8], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                for group in pl.range(block, (rows + row_tile - 1) // row_tile, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    source_valid = pl.load(x, [row_start, 0], [row_tile, padded_hidden], valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec)
                    source_storage = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                    source_fp32 = pl.tile.cast(source_storage, target_type=pl.FP32)
                    source_segments = pl.tile.reshape(source_fp32, [8, segment_width])
                    squared = pl.tile.mul(source_segments, source_segments)
                    reduction_scratch = pl.tile.create([8, segment_width], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    partial_column = pl.tile.row_sum(squared, reduction_scratch)
                    square_sum = pl.tile.reshape(partial_column, [1, 8])
                    gathered_1 = pl.tile.gather(square_sum, indices_1, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_1)
                    gathered_2 = pl.tile.gather(square_sum, indices_2, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_2)
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [8, 1])
                    output_source = pl.tile.reshape(pl.tile.cast(source_storage, target_type=pl.FP32), [8, segment_width])
                    normalized = pl.tile.row_expand_mul(output_source, inverse_column)
                    logical_output = pl.tile.reshape(normalized, [row_tile, padded_hidden])
                    stored = pl.tile.cast(logical_output, target_type=storage_type)
                    stored_valid = pl.tile.set_validshape(stored, valid_rows, hidden)
                    y = pl.store(stored_valid, [row_start, 0], y)
            return y

    return SegmentedRMSNorm2


def _make_segmented_4(rows, hidden, row_tile, padded_hidden, blocks, storage_type):
    segment_width = padded_hidden // 2

    @pl.program
    class SegmentedRMSNorm4:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="segment_rows"):
                indices_1 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_1, [0, 0], pl.const(1, pl.INT32))
                pl.tile.write(indices_1, [0, 1], pl.const(0, pl.INT32))
                pl.tile.write(indices_1, [0, 2], pl.const(3, pl.INT32))
                pl.tile.write(indices_1, [0, 3], pl.const(2, pl.INT32))
                pl.tile.write(indices_1, [0, 4], pl.const(5, pl.INT32))
                pl.tile.write(indices_1, [0, 5], pl.const(4, pl.INT32))
                pl.tile.write(indices_1, [0, 6], pl.const(7, pl.INT32))
                pl.tile.write(indices_1, [0, 7], pl.const(6, pl.INT32))
                gather_scratch = pl.tile.create([1, 8], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                for group in pl.range(block, (rows + row_tile - 1) // row_tile, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    source_valid = pl.load(x, [row_start, 0], [row_tile, padded_hidden], valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec)
                    source_storage = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                    source_fp32 = pl.tile.cast(source_storage, target_type=pl.FP32)
                    source_segments = pl.tile.reshape(source_fp32, [8, segment_width])
                    squared = pl.tile.mul(source_segments, source_segments)
                    reduction_scratch = pl.tile.create([8, segment_width], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    partial_column = pl.tile.row_sum(squared, reduction_scratch)
                    square_sum = pl.tile.reshape(partial_column, [1, 8])
                    gathered_1 = pl.tile.gather(square_sum, indices_1, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_1)
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [8, 1])
                    output_source = pl.tile.reshape(pl.tile.cast(source_storage, target_type=pl.FP32), [8, segment_width])
                    normalized = pl.tile.row_expand_mul(output_source, inverse_column)
                    logical_output = pl.tile.reshape(normalized, [row_tile, padded_hidden])
                    stored = pl.tile.cast(logical_output, target_type=storage_type)
                    stored_valid = pl.tile.set_validshape(stored, valid_rows, hidden)
                    y = pl.store(stored_valid, [row_start, 0], y)
            return y

    return SegmentedRMSNorm4



def _make_native_narrow(rows, hidden, row_tile, padded_hidden, blocks, storage_type):
    @pl.program
    class NativeNarrowRMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="native_narrow_rows"):
                for group in pl.range(block, (rows + row_tile - 1) // row_tile, blocks):
                    row_start = group * row_tile
                    valid_rows = pl.min(row_tile, rows - row_start)
                    source_valid = pl.load(x, [row_start, 0], [row_tile, padded_hidden], valid_shape=[valid_rows, hidden], target_memory=pl.MemorySpace.Vec)
                    source_storage = pl.tile.fillpad_inplace(source_valid, pad_value=pl.PadValue.zero)
                    source_fp32 = pl.tile.cast(source_storage, target_type=pl.FP32)
                    squared = pl.tile.mul(source_fp32, source_fp32)
                    reduction_scratch = pl.tile.create([row_tile, padded_hidden], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    square_column = pl.tile.row_sum(squared, reduction_scratch)
                    square_sum = pl.tile.reshape(square_column, [1, row_tile])
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, row_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [row_tile, 1])
                    output_source = pl.tile.cast(source_storage, target_type=pl.FP32)
                    normalized = pl.tile.row_expand_mul(output_source, inverse_column)
                    stored = pl.tile.cast(normalized, target_type=storage_type)
                    stored_valid = pl.tile.set_validshape(stored, valid_rows, hidden)
                    y = pl.store(stored_valid, [row_start, 0], y)
            return y

    return NativeNarrowRMSNorm



RESIDENT_PREFIX_COLS = 16384
FRINGE_TILE_COLS = 1024
HYBRID_MIN = 17921
HYBRID_MAX = 18432


def geometry(shape):
    layout = _donor_geometry(shape)
    hidden = layout["hidden"]
    prefix_cols = RESIDENT_PREFIX_COLS
    fringe_cols = FRINGE_TILE_COLS
    prefix_phase_bytes = 10 * prefix_cols
    fringe_phase_bytes = 2 * prefix_cols + 10 * fringe_cols
    output_phase_bytes = 8 * prefix_cols
    hybrid_bytes = max(prefix_phase_bytes, fringe_phase_bytes, output_phase_bytes) + 16 * 1024
    donor_mode = layout["mode"]
    eligible = (shape["dtype"] == "bf16" and donor_mode == "streaming"
                and layout["route_reason"] == "residency_fallback"
                and HYBRID_MIN <= hidden <= HYBRID_MAX
                and prefix_cols < hidden <= 2 * prefix_cols)
    layout.update(donor_mode=donor_mode, donor_row_tile=layout["row_tile"],
                  requested_prefix_cols=prefix_cols, requested_fringe_cols=fringe_cols,
                  hybrid_activation=[HYBRID_MIN, HYBRID_MAX], hybrid_bytes=hybrid_bytes,
                  prefix_phase_bytes=prefix_phase_bytes, fringe_phase_bytes=fringe_phase_bytes,
                  output_phase_bytes=output_phase_bytes, hybrid_reserve_bytes=16 * 1024,
                  selected_prefix_cols=None, selected_fringe_cols=None)
    if eligible and hybrid_bytes <= VEC_LIMIT_BYTES:
        fringe_width = hidden - prefix_cols
        layout.update(mode="resident_prefix_streamed_fringe", row_tile=1,
                      donor_col_tile=layout["col_tile"], donor_cached_width=layout["cached_width"],
                      donor_cache_bytes=layout["cache_bytes"],
                      col_tile=fringe_cols, cached_width=prefix_cols, cache_bytes=2 * prefix_cols,
                      prefix_cols=prefix_cols, fringe_cols=fringe_cols,
                      selected_prefix_cols=prefix_cols, selected_fringe_cols=fringe_cols,
                      prefix_valid_cols=prefix_cols, fringe_valid_cols=fringe_width,
                      fringe_tiles=(fringe_width + fringe_cols - 1) // fringe_cols,
                      final_fringe_valid_cols=(fringe_width - 1) % fringe_cols + 1,
                      blocks=min(DISPATCH_BLOCKS, layout["rows"]),
                      selected_output_pipeline=0, selected_reduction_mode=-1,
                      route_reason="bounded_prefix_streamed_fringe")
    elif eligible:
        layout["route_reason"] = "prefix_capacity_donor_fallback"
    return layout


def _make_resident_prefix(rows, hidden, prefix_cols, fringe_cols, blocks, storage_type):
    prefix_segment_width = prefix_cols // 8
    fringe_segment_width = fringe_cols // 8

    @pl.program
    class ResidentPrefixRMSNorm:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], storage_type],
            y: pl.Out[pl.Tensor[[rows, hidden], storage_type]],
        ):
            for block in pl.spmd(blocks, name_hint="prefix_fringe_rows"):
                indices_1 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                indices_2 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                indices_4 = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(indices_1, [0, 0], pl.const(1, pl.INT32))
                pl.tile.write(indices_1, [0, 1], pl.const(0, pl.INT32))
                pl.tile.write(indices_1, [0, 2], pl.const(3, pl.INT32))
                pl.tile.write(indices_1, [0, 3], pl.const(2, pl.INT32))
                pl.tile.write(indices_1, [0, 4], pl.const(5, pl.INT32))
                pl.tile.write(indices_1, [0, 5], pl.const(4, pl.INT32))
                pl.tile.write(indices_1, [0, 6], pl.const(7, pl.INT32))
                pl.tile.write(indices_1, [0, 7], pl.const(6, pl.INT32))
                pl.tile.write(indices_2, [0, 0], pl.const(2, pl.INT32))
                pl.tile.write(indices_2, [0, 1], pl.const(3, pl.INT32))
                pl.tile.write(indices_2, [0, 2], pl.const(0, pl.INT32))
                pl.tile.write(indices_2, [0, 3], pl.const(1, pl.INT32))
                pl.tile.write(indices_2, [0, 4], pl.const(6, pl.INT32))
                pl.tile.write(indices_2, [0, 5], pl.const(7, pl.INT32))
                pl.tile.write(indices_2, [0, 6], pl.const(4, pl.INT32))
                pl.tile.write(indices_2, [0, 7], pl.const(5, pl.INT32))
                pl.tile.write(indices_4, [0, 0], pl.const(4, pl.INT32))
                pl.tile.write(indices_4, [0, 1], pl.const(5, pl.INT32))
                pl.tile.write(indices_4, [0, 2], pl.const(6, pl.INT32))
                pl.tile.write(indices_4, [0, 3], pl.const(7, pl.INT32))
                pl.tile.write(indices_4, [0, 4], pl.const(0, pl.INT32))
                pl.tile.write(indices_4, [0, 5], pl.const(1, pl.INT32))
                pl.tile.write(indices_4, [0, 6], pl.const(2, pl.INT32))
                pl.tile.write(indices_4, [0, 7], pl.const(3, pl.INT32))
                gather_scratch = pl.tile.create([1, 8], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                for row_start in pl.range(block, rows, blocks):
                    prefix_storage = pl.load(x, [row_start, 0], [1, prefix_cols], target_memory=pl.MemorySpace.Vec)
                    prefix_fp32 = pl.tile.cast(prefix_storage, target_type=pl.FP32)
                    prefix_segments = pl.tile.reshape(prefix_fp32, [8, prefix_segment_width])
                    prefix_squared = pl.tile.mul(prefix_segments, prefix_segments)
                    prefix_scratch = pl.tile.create([8, prefix_segment_width], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    prefix_partial = pl.tile.row_sum(prefix_squared, prefix_scratch)
                    square_sum = pl.tile.reshape(prefix_partial, [1, 8])
                    for col_start in pl.range(prefix_cols, hidden, fringe_cols):
                        valid_cols = pl.min(fringe_cols, hidden - col_start)
                        fringe_valid = pl.load(x, [row_start, col_start], [1, fringe_cols], valid_shape=[1, valid_cols], target_memory=pl.MemorySpace.Vec)
                        fringe_padded = pl.tile.fillpad_inplace(fringe_valid, pad_value=pl.PadValue.zero)
                        fringe_storage = pl.tile.set_validshape(fringe_padded, 1, fringe_cols)
                        fringe_fp32 = pl.tile.cast(fringe_storage, target_type=pl.FP32)
                        fringe_segments = pl.tile.reshape(fringe_fp32, [8, fringe_segment_width])
                        fringe_squared = pl.tile.mul(fringe_segments, fringe_segments)
                        fringe_scratch = pl.tile.create([8, fringe_segment_width], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        fringe_partial = pl.tile.row_sum(fringe_squared, fringe_scratch)
                        fringe_row = pl.tile.reshape(fringe_partial, [1, 8])
                        square_sum = pl.tile.add(square_sum, fringe_row)
                    gathered_1 = pl.tile.gather(square_sum, indices_1, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_1)
                    gathered_2 = pl.tile.gather(square_sum, indices_2, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_2)
                    gathered_4 = pl.tile.gather(square_sum, indices_4, gather_scratch)
                    square_sum = pl.tile.add(square_sum, gathered_4)
                    mean_square = pl.tile.muls(square_sum, 1.0 / hidden)
                    stabilized = pl.tile.adds(mean_square, 1e-6)
                    rsqrt_scratch = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    inverse_row = pl.tile.rsqrt(stabilized, tmp=rsqrt_scratch)
                    inverse_column = pl.tile.reshape(inverse_row, [8, 1])
                    prefix_output = pl.tile.cast(prefix_storage, target_type=pl.FP32)
                    prefix_output_segments = pl.tile.reshape(prefix_output, [8, prefix_segment_width])
                    prefix_normalized = pl.tile.row_expand_mul(prefix_output_segments, inverse_column)
                    prefix_logical = pl.tile.reshape(prefix_normalized, [1, prefix_cols])
                    prefix_stored = pl.tile.cast(prefix_logical, target_type=storage_type)
                    y = pl.store(prefix_stored, [row_start, 0], y)
                    for col_start in pl.range(prefix_cols, hidden, fringe_cols):
                        valid_cols = pl.min(fringe_cols, hidden - col_start)
                        output_valid = pl.load(x, [row_start, col_start], [1, fringe_cols], valid_shape=[1, valid_cols], target_memory=pl.MemorySpace.Vec)
                        output_padded = pl.tile.fillpad_inplace(output_valid, pad_value=pl.PadValue.zero)
                        output_storage = pl.tile.set_validshape(output_padded, 1, fringe_cols)
                        output_fp32 = pl.tile.cast(output_storage, target_type=pl.FP32)
                        output_segments = pl.tile.reshape(output_fp32, [8, fringe_segment_width])
                        normalized = pl.tile.row_expand_mul(output_segments, inverse_column)
                        output_logical = pl.tile.reshape(normalized, [1, fringe_cols])
                        stored = pl.tile.cast(output_logical, target_type=storage_type)
                        stored_valid = pl.tile.set_validshape(stored, 1, valid_cols)
                        y = pl.store(stored_valid, [row_start, col_start], y)
            return y

    return ResidentPrefixRMSNorm


def build(shape):
    import torch
    from golden import TensorSpec

    layout = geometry(shape)
    rows = layout["rows"]
    hidden = layout["hidden"]
    storage_type = {"fp32": pl.FP32, "bf16": pl.BF16}[shape["dtype"]]
    factories = {
        ("scalar", "bf16"): _make_scalar_bf16,
        ("scalar", "fp32"): _make_scalar_fp32,
        ("unit", "bf16"): _make_unit_bf16,
        ("unit", "fp32"): _make_unit_fp32,
        ("compact", "bf16"): _make_compact_bf16,
        ("streaming", "bf16"): _make_streaming_bf16,
        ("compact", "fp32"): _make_compact_fp32,
        ("streaming", "fp32"): _make_streaming_fp32,
    }
    if layout["mode"] == "resident_prefix_streamed_fringe":
        fn = _make_resident_prefix(rows, hidden, layout["prefix_cols"],
                                   layout["fringe_cols"], layout["blocks"], storage_type)
    elif layout["mode"] == "native_narrow":
        fn = _make_native_narrow(rows, hidden, layout["row_tile"], layout["padded_hidden"],
                                 layout["blocks"], storage_type)
    elif layout["mode"] == "segmented":
        factory = {1: _make_segmented_1, 2: _make_segmented_2, 4: _make_segmented_4}[layout["row_tile"]]
        fn = factory(rows, hidden, layout["row_tile"], layout["padded_hidden"],
                     layout["blocks"], storage_type)
    else:
        factory = factories[(layout["mode"], shape["dtype"])]
        fn = factory(rows, hidden, layout["row_tile"], layout["col_tile"],
                     layout["cached_width"], layout["blocks"], storage_type)
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[shape["dtype"]]
    specs = [
        TensorSpec("x", [rows, hidden], torch_dtype, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], torch_dtype),
    ]

    def golden_fn(tensors):
        import torch

        x = tensors["x"]
        x32 = x.float()
        y32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y = y32.to(torch_dtype)
        tensors["y"].copy_(y)

    if shape["dtype"] == "bf16":
        return fn, specs, golden_fn, 1e-2, 1e-2
    return fn, specs, golden_fn, 1e-4, 1e-4

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
