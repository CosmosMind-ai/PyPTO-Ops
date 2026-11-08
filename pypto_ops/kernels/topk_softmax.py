# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capacity-scaled FP32 softmax with chronological fixed-length heap selection."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}

ROWS_PER_TASK = 1
BLOCK_WIDTH = 32
RESIDENT_REQUESTED_BLOCK_WIDTH = 32
RESIDENT_BLOCK_WIDTH = RESIDENT_REQUESTED_BLOCK_WIDTH
RESIDENT_TAIL_WIDTH = 128
RESIDENT_SUMMARY_BLOCKS = 8
RESIDENT_SUMMARY_PREFIX_WIDTH = 128
CHUNK_WIDTH = 128
RESIDENT_CAPACITY = 8192
NARROW_DEPTH = 14
INPUT_IS_BF16 = True
PARENT_CHILDREN = 8
PARENT_WIDTH = PARENT_CHILDREN * RESIDENT_REQUESTED_BLOCK_WIDTH
PARENT_SUMMARY_PREFIX_WIDTH = 0
SUFFIX_SUMMARY_PREFIX_WIDTH = 0
SUFFIX_SUMMARY_TILE_WIDTH = 256
SUFFIX_SUMMARY_OFFSET = 0

SUMMARY_RESIDUAL_WIDTH = 128
SUMMARY_RESIDUAL_OFFSET = 0

@pl.jit
def resident(logits: pl.Tensor, colidx: pl.Tensor, p: pl.Out[pl.Tensor], topk_w: pl.Out[pl.Tensor], topk_idx: pl.Out[pl.Tensor]):
    rows, cols = logits.shape
    for row_block in pl.spmd((rows + ROWS_PER_TASK - 1) // ROWS_PER_TASK, name_hint='capacity_resident_gating'):
        row_base = row_block * ROWS_PER_TASK
        for row_offset in pl.range(ROWS_PER_TASK):
            row = row_base + row_offset
            if row < rows:
                input_values = pl.slice(logits, [1, cols], [row, 0])
                values = pl.cast(input_values, pl.FP32)
                summary_logits = pl.slice(values, [1, cols // (RESIDENT_BLOCK_WIDTH * 8) * (RESIDENT_BLOCK_WIDTH * 8)], [0, 0])
                summary_logit_regions = pl.reshape(summary_logits, [(cols // (RESIDENT_BLOCK_WIDTH * 8) * (RESIDENT_BLOCK_WIDTH * 8)) // RESIDENT_BLOCK_WIDTH, RESIDENT_BLOCK_WIDTH])
                summary_logit_maxima = pl.row_max(summary_logit_regions)
                summary_maximum_lanes = pl.col_max(pl.reshape(summary_logit_maxima, [(cols // (RESIDENT_BLOCK_WIDTH * 8) * (RESIDENT_BLOCK_WIDTH * 8)) // RESIDENT_BLOCK_WIDTH // 8, 8]))
                maximum = pl.read(summary_maximum_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(summary_maximum_lanes, [0, lane])
                    if lane_value > maximum:
                        maximum = lane_value
                if cols > cols // (RESIDENT_BLOCK_WIDTH * 8) * (RESIDENT_BLOCK_WIDTH * 8):
                    residual_logits = pl.slice(values, [1, SUMMARY_RESIDUAL_WIDTH], [0, SUMMARY_RESIDUAL_OFFSET])
                    residual_maximum_lanes = pl.col_max(pl.reshape(residual_logits, [SUMMARY_RESIDUAL_WIDTH // 8, 8]))
                    for lane in pl.unroll(8):
                        lane_value = pl.read(residual_maximum_lanes, [0, lane])
                        if lane_value > maximum:
                            maximum = lane_value
                exponentials = pl.exp(pl.sub(values, maximum))
                denominator_lanes = pl.col_sum(pl.reshape(exponentials, [cols // 8, 8]))
                denominator = pl.read(denominator_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(denominator_lanes, [0, lane])
                    denominator = denominator + lane_value
                probabilities = pl.div(exponentials, denominator)
                bitmap_weights = pl.full([1, 16], dtype=pl.FP32, value=0.0)
                pl.write(bitmap_weights, [0, 0], 1.0)
                pl.write(bitmap_weights, [0, 1], 2.0)
                pl.write(bitmap_weights, [0, 2], 4.0)
                pl.write(bitmap_weights, [0, 3], 8.0)
                pl.write(bitmap_weights, [0, 4], 16.0)
                pl.write(bitmap_weights, [0, 5], 32.0)
                pl.write(bitmap_weights, [0, 6], 64.0)
                pl.write(bitmap_weights, [0, 7], 128.0)
                pl.write(bitmap_weights, [0, 8], 256.0)
                pl.write(bitmap_weights, [0, 9], 512.0)
                pl.write(bitmap_weights, [0, 10], 1024.0)
                pl.write(bitmap_weights, [0, 11], 2048.0)
                pl.write(bitmap_weights, [0, 12], 4096.0)
                pl.write(bitmap_weights, [0, 13], 8192.0)
                pl.write(bitmap_weights, [0, 14], 16384.0)
                pl.write(bitmap_weights, [0, 15], 32768.0)
                if cols < 512:
                    columns = pl.slice(colidx, [1, cols], [0, 0])
                    partition_values = pl.add(probabilities, 0.0)
                    partition_indices = pl.cast(columns, pl.INT32)
                    first = 0
                    last = cols
                    depth_limit = NARROW_DEPTH
                    for bounded_step_8 in pl.range(NARROW_DEPTH + 1):
                        if last - first > 3:
                            if depth_limit == 0:
                                heap_length = 8 - first
                                selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                                selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                                for slot in pl.range(heap_length):
                                    initial_value = pl.read(partition_values, [0, first + slot])
                                    initial_index = pl.read(partition_indices, [0, first + slot])
                                    pl.write(selected_values, [0, slot], initial_value)
                                    pl.write(selected_indices, [0, slot], initial_index)
                                if heap_length > 1:
                                    for parent_step in pl.range((heap_length - 2) // 2 + 1):
                                        hole = (heap_length - 2) // 2 - parent_step
                                        incoming_value = pl.read(selected_values, [0, hole])
                                        incoming_index = pl.read(selected_indices, [0, hole])
                                        hole_adjust_1 = hole
                                        top_adjust_1 = hole_adjust_1
                                        child_adjust_1 = hole_adjust_1
                                        for bounded_step_1 in pl.range(3):
                                            if child_adjust_1 < (heap_length - 1) // 2:
                                                child_adjust_1 = 2 * (child_adjust_1 + 1)
                                                right_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                left_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                                if right_value_adjust_1 > left_value_adjust_1:
                                                    child_adjust_1 = child_adjust_1 - 1
                                                moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1])
                                                pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                hole_adjust_1 = child_adjust_1
                                        if heap_length % 2 == 0 and child_adjust_1 == (heap_length - 2) // 2:
                                            child_adjust_1 = 2 * (child_adjust_1 + 1)
                                            moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                            moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1 - 1])
                                            pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                            pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                            hole_adjust_1 = child_adjust_1 - 1
                                        for bounded_step_2 in pl.range(3):
                                            if hole_adjust_1 > top_adjust_1:
                                                parent_adjust_1 = (hole_adjust_1 - 1) // 2
                                                parent_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                if not parent_value_adjust_1 <= incoming_value:
                                                    moving_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                    moving_index_adjust_1 = pl.read(selected_indices, [0, parent_adjust_1])
                                                    pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                    pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                    hole_adjust_1 = parent_adjust_1
                                        pl.write(selected_values, [0, hole_adjust_1], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_1], incoming_index)
                                for scan_column in pl.range(8, last):
                                    incoming_value = pl.read(partition_values, [0, scan_column])
                                    root = pl.read(selected_values, [0, 0])
                                    if incoming_value > root:
                                        incoming_index = pl.read(partition_indices, [0, scan_column])
                                        hole = 0
                                        hole_adjust_2 = hole
                                        top_adjust_2 = hole_adjust_2
                                        child_adjust_2 = hole_adjust_2
                                        for bounded_step_3 in pl.range(3):
                                            if child_adjust_2 < (heap_length - 1) // 2:
                                                child_adjust_2 = 2 * (child_adjust_2 + 1)
                                                right_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                left_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                                if right_value_adjust_2 > left_value_adjust_2:
                                                    child_adjust_2 = child_adjust_2 - 1
                                                moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2])
                                                pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                hole_adjust_2 = child_adjust_2
                                        if heap_length % 2 == 0 and child_adjust_2 == (heap_length - 2) // 2:
                                            child_adjust_2 = 2 * (child_adjust_2 + 1)
                                            moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                            moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2 - 1])
                                            pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                            pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                            hole_adjust_2 = child_adjust_2 - 1
                                        for bounded_step_4 in pl.range(3):
                                            if hole_adjust_2 > top_adjust_2:
                                                parent_adjust_2 = (hole_adjust_2 - 1) // 2
                                                parent_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                if not parent_value_adjust_2 <= incoming_value:
                                                    moving_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                    moving_index_adjust_2 = pl.read(selected_indices, [0, parent_adjust_2])
                                                    pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                    pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                    hole_adjust_2 = parent_adjust_2
                                        pl.write(selected_values, [0, hole_adjust_2], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_2], incoming_index)
                                for slot in pl.range(heap_length):
                                    copied_value = pl.read(selected_values, [0, slot])
                                    copied_index = pl.read(selected_indices, [0, slot])
                                    pl.write(partition_values, [0, first + slot], copied_value)
                                    pl.write(partition_indices, [0, first + slot], copied_index)
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, 7])
                                moving_index = pl.read(partition_indices, [0, 7])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, 7], swap_value)
                                pl.write(partition_indices, [0, 7], swap_index)
                                first = last
                            else:
                                depth_limit = depth_limit - 1
                                median_left = first + 1
                                median_middle = first + (last - first) // 2
                                median_right = last - 1
                                left_value = pl.read(partition_values, [0, median_left])
                                middle_value = pl.read(partition_values, [0, median_middle])
                                right_value = pl.read(partition_values, [0, median_right])
                                pivot_column = median_middle
                                if left_value > middle_value:
                                    if middle_value > right_value:
                                        pivot_column = median_middle
                                    elif left_value > right_value:
                                        pivot_column = median_right
                                    else:
                                        pivot_column = median_left
                                elif left_value > right_value:
                                    pivot_column = median_left
                                elif middle_value > right_value:
                                    pivot_column = median_right
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, pivot_column])
                                moving_index = pl.read(partition_indices, [0, pivot_column])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, pivot_column], swap_value)
                                pl.write(partition_indices, [0, pivot_column], swap_index)
                                pivot_value = pl.read(partition_values, [0, first])
                                partition_left = first + 1
                                partition_right = last
                                partition_active = True
                                for bounded_step_7 in pl.range(cols + 8):
                                    if partition_active:
                                        probe_left = pl.read(partition_values, [0, partition_left])
                                        for bounded_step_5 in pl.range(cols + 8):
                                            if probe_left > pivot_value:
                                                partition_left = partition_left + 1
                                                probe_left = pl.read(partition_values, [0, partition_left])
                                        partition_right = partition_right - 1
                                        probe_right = pl.read(partition_values, [0, partition_right])
                                        for bounded_step_6 in pl.range(cols + 8):
                                            if pivot_value > probe_right:
                                                partition_right = partition_right - 1
                                                probe_right = pl.read(partition_values, [0, partition_right])
                                        if partition_left >= partition_right:
                                            partition_active = False
                                        else:
                                            swap_value = pl.read(partition_values, [0, partition_left])
                                            swap_index = pl.read(partition_indices, [0, partition_left])
                                            moving_value = pl.read(partition_values, [0, partition_right])
                                            moving_index = pl.read(partition_indices, [0, partition_right])
                                            pl.write(partition_values, [0, partition_left], moving_value)
                                            pl.write(partition_indices, [0, partition_left], moving_index)
                                            pl.write(partition_values, [0, partition_right], swap_value)
                                            pl.write(partition_indices, [0, partition_right], swap_index)
                                            partition_left = partition_left + 1
                                if partition_left <= 7:
                                    first = partition_left
                                else:
                                    last = partition_left
                    for insertion_column in pl.range(first + 1, last):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_9 in pl.range(cols + 8):
                            if insertion_hole > first:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    for insertion_column in pl.range(0 + 1, 7):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_10 in pl.range(cols + 8):
                            if insertion_hole > 0:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    for slot in pl.range(8):
                        copied_value = pl.read(partition_values, [0, slot])
                        copied_index = pl.read(partition_indices, [0, slot])
                        pl.write(selected_values, [0, slot], copied_value)
                        pl.write(selected_indices, [0, slot], copied_index)
                else:
                    heap_value_0 = pl.read(probabilities, [0, 0])
                    heap_index_0 = pl.cast(0, pl.INT32)
                    heap_value_1 = pl.read(probabilities, [0, 1])
                    heap_index_1 = pl.cast(1, pl.INT32)
                    heap_value_2 = pl.read(probabilities, [0, 2])
                    heap_index_2 = pl.cast(2, pl.INT32)
                    heap_value_3 = pl.read(probabilities, [0, 3])
                    heap_index_3 = pl.cast(3, pl.INT32)
                    heap_value_4 = pl.read(probabilities, [0, 4])
                    heap_index_4 = pl.cast(4, pl.INT32)
                    heap_value_5 = pl.read(probabilities, [0, 5])
                    heap_index_5 = pl.cast(5, pl.INT32)
                    heap_value_6 = pl.read(probabilities, [0, 6])
                    heap_index_6 = pl.cast(6, pl.INT32)
                    heap_value_7 = pl.read(probabilities, [0, 7])
                    heap_index_7 = pl.cast(7, pl.INT32)
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_7
                    heap_index_3 = heap_index_7
                    if heap_value_3 > incoming_value:
                        heap_value_7 = heap_value_3
                        heap_index_7 = heap_index_3
                        heap_value_3 = incoming_value
                        heap_index_3 = incoming_index
                    else:
                        heap_value_7 = incoming_value
                        heap_index_7 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    if heap_value_6 > heap_value_5:
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    else:
                        heap_value_2 = heap_value_6
                        heap_index_2 = heap_index_6
                        if heap_value_2 > incoming_value:
                            heap_value_6 = heap_value_2
                            heap_index_6 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_6 = incoming_value
                            heap_index_6 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    if heap_value_4 > heap_value_3:
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        heap_value_3 = heap_value_7
                        heap_index_3 = heap_index_7
                        if heap_value_3 > incoming_value:
                            heap_value_7 = heap_value_3
                            heap_index_7 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_7 = incoming_value
                            heap_index_7 = incoming_index
                    else:
                        heap_value_1 = heap_value_4
                        heap_index_1 = heap_index_4
                        if heap_value_1 > incoming_value:
                            heap_value_4 = heap_value_1
                            heap_index_4 = heap_index_1
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                        else:
                            heap_value_4 = incoming_value
                            heap_index_4 = incoming_index
                    incoming_value = heap_value_0
                    incoming_index = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            heap_value_3 = heap_value_7
                            heap_index_3 = heap_index_7
                            if heap_value_3 > incoming_value:
                                heap_value_7 = heap_value_3
                                heap_index_7 = heap_index_3
                                if heap_value_1 > incoming_value:
                                    heap_value_3 = heap_value_1
                                    heap_index_3 = heap_index_1
                                    if heap_value_0 > incoming_value:
                                        heap_value_1 = heap_value_0
                                        heap_index_1 = heap_index_0
                                        heap_value_0 = incoming_value
                                        heap_index_0 = incoming_index
                                    else:
                                        heap_value_1 = incoming_value
                                        heap_index_1 = incoming_index
                                else:
                                    heap_value_3 = incoming_value
                                    heap_index_3 = incoming_index
                            else:
                                heap_value_7 = incoming_value
                                heap_index_7 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    path_active = False
                    path_hole = 0
                    path_second = 0
                    path_second_value = heap_value_0
                    path_second_index = heap_index_0
                    block_max = heap_value_0
                    if PARENT_SUMMARY_PREFIX_WIDTH > 0:
                        parent_prefix = pl.slice(probabilities, [1, PARENT_SUMMARY_PREFIX_WIDTH], [0, 0])
                        parent_values = pl.reshape(parent_prefix, [PARENT_SUMMARY_PREFIX_WIDTH // PARENT_WIDTH, PARENT_WIDTH])
                        parent_maxima = pl.row_max(parent_values)
                        for parent in pl.range(PARENT_SUMMARY_PREFIX_WIDTH // PARENT_WIDTH):
                            parent_max = pl.read(parent_maxima, [parent, 0])
                            if parent_max > heap_value_0:
                                parent_probabilities = pl.slice(probabilities, [1, PARENT_WIDTH], [0, parent * PARENT_WIDTH])
                                packet_mask = pl.cmp(parent_probabilities, heap_value_0, cmp_type=4)
                                packet_halves = pl.reshape(packet_mask, [PARENT_WIDTH // 16, 16])
                                packet_weighted = pl.col_expand_mul(packet_halves, bitmap_weights)
                                packet_words = pl.cast(pl.row_sum(packet_weighted), pl.INT32)
                                child_values = pl.reshape(parent_probabilities, [PARENT_CHILDREN, RESIDENT_BLOCK_WIDTH])
                                child_maxima = pl.row_max(child_values)
                                for child in pl.range(PARENT_CHILDREN):
                                    block = parent * PARENT_CHILDREN + child
                                    column = block * RESIDENT_BLOCK_WIDTH
                                    block_max = pl.read(child_maxima, [child, 0])
                                    if block_max > heap_value_0:
                                        leaf_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                                        for bitmap_half in pl.range(2):
                                            bitmap_word = pl.read(packet_words, [child * 2 + bitmap_half, 0])
                                            bitmap_bit = 0
                                            for bitmap_iteration in pl.range(16):
                                                if bitmap_word == 0:
                                                    break
                                                block_offset = bitmap_half * 16 + bitmap_bit
                                                if bitmap_word & 1 != 0:
                                                    candidate_column = column + block_offset
                                                    if candidate_column >= 8:
                                                        incoming_value = pl.read(leaf_probabilities, [0, block_offset])
                                                        root_resident_0_path_first = 0
                                                        root_resident_0_path_first_value = 0.0
                                                        root_resident_0_path_first_index = pl.cast(0, pl.INT32)
                                                        root_resident_0_path_second = 0
                                                        root_resident_0_path_second_value = 0.0
                                                        root_resident_0_path_second_index = pl.cast(0, pl.INT32)
                                                        if incoming_value > heap_value_0:
                                                            incoming_index = pl.cast(candidate_column, pl.INT32)
                                                            root_resident_0_path_first = 2
                                                            root_resident_0_path_first_value = heap_value_2
                                                            root_resident_0_path_first_index = heap_index_2
                                                            if heap_value_2 > heap_value_1:
                                                                root_resident_0_path_first = 1
                                                                root_resident_0_path_first_value = heap_value_1
                                                                root_resident_0_path_first_index = heap_index_1
                                                            if root_resident_0_path_first_value > incoming_value:
                                                                heap_value_0 = incoming_value
                                                                heap_index_0 = incoming_index
                                                            else:
                                                                heap_value_0 = root_resident_0_path_first_value
                                                                heap_index_0 = root_resident_0_path_first_index
                                                                root_resident_0_path_second = 6
                                                                root_resident_0_path_second_value = heap_value_6
                                                                root_resident_0_path_second_index = heap_index_6
                                                                if root_resident_0_path_first == 1:
                                                                    root_resident_0_path_second = 4
                                                                    root_resident_0_path_second_value = heap_value_4
                                                                    root_resident_0_path_second_index = heap_index_4
                                                                    if heap_value_4 > heap_value_3:
                                                                        root_resident_0_path_second = 3
                                                                        root_resident_0_path_second_value = heap_value_3
                                                                        root_resident_0_path_second_index = heap_index_3
                                                                elif heap_value_6 > heap_value_5:
                                                                    root_resident_0_path_second = 5
                                                                    root_resident_0_path_second_value = heap_value_5
                                                                    root_resident_0_path_second_index = heap_index_5
                                                                if root_resident_0_path_second_value > incoming_value:
                                                                    if root_resident_0_path_first == 1:
                                                                        heap_value_1 = incoming_value
                                                                        heap_index_1 = incoming_index
                                                                    else:
                                                                        heap_value_2 = incoming_value
                                                                        heap_index_2 = incoming_index
                                                                else:
                                                                    if root_resident_0_path_first == 1:
                                                                        heap_value_1 = root_resident_0_path_second_value
                                                                        heap_index_1 = root_resident_0_path_second_index
                                                                    else:
                                                                        heap_value_2 = root_resident_0_path_second_value
                                                                        heap_index_2 = root_resident_0_path_second_index
                                                                    if root_resident_0_path_second == 3:
                                                                        if heap_value_7 > incoming_value:
                                                                            heap_value_3 = incoming_value
                                                                            heap_index_3 = incoming_index
                                                                        else:
                                                                            heap_value_3 = heap_value_7
                                                                            heap_index_3 = heap_index_7
                                                                            heap_value_7 = incoming_value
                                                                            heap_index_7 = incoming_index
                                                                    elif root_resident_0_path_second == 4:
                                                                        heap_value_4 = incoming_value
                                                                        heap_index_4 = incoming_index
                                                                    elif root_resident_0_path_second == 5:
                                                                        heap_value_5 = incoming_value
                                                                        heap_index_5 = incoming_index
                                                                    else:
                                                                        heap_value_6 = incoming_value
                                                                        heap_index_6 = incoming_index
                                                bitmap_word = pl.cast(bitmap_word >> 1, pl.INT32)
                                                bitmap_bit = bitmap_bit + 1
                    if SUFFIX_SUMMARY_PREFIX_WIDTH > 0:
                        suffix_prefix = pl.slice(probabilities, [1, SUFFIX_SUMMARY_TILE_WIDTH], [0, SUFFIX_SUMMARY_OFFSET])
                        suffix_values = pl.reshape(suffix_prefix, [SUFFIX_SUMMARY_TILE_WIDTH // RESIDENT_BLOCK_WIDTH, RESIDENT_BLOCK_WIDTH])
                        suffix_maxima = pl.row_max(suffix_values)
                        for block in pl.range(PARENT_SUMMARY_PREFIX_WIDTH // RESIDENT_BLOCK_WIDTH, (PARENT_SUMMARY_PREFIX_WIDTH + SUFFIX_SUMMARY_PREFIX_WIDTH) // RESIDENT_BLOCK_WIDTH):
                            column = block * RESIDENT_BLOCK_WIDTH
                            block_max = pl.read(suffix_maxima, [block - PARENT_SUMMARY_PREFIX_WIDTH // RESIDENT_BLOCK_WIDTH, 0])
                            if block_max > heap_value_0:
                                leaf_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                                for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                    candidate_column = column + block_offset
                                    if candidate_column >= 8:
                                        incoming_value = pl.read(leaf_probabilities, [0, block_offset])
                                        root_resident_1_path_first = 0
                                        root_resident_1_path_first_value = 0.0
                                        root_resident_1_path_first_index = pl.cast(0, pl.INT32)
                                        root_resident_1_path_second = 0
                                        root_resident_1_path_second_value = 0.0
                                        root_resident_1_path_second_index = pl.cast(0, pl.INT32)
                                        if incoming_value > heap_value_0:
                                            incoming_index = pl.cast(candidate_column, pl.INT32)
                                            root_resident_1_path_first = 2
                                            root_resident_1_path_first_value = heap_value_2
                                            root_resident_1_path_first_index = heap_index_2
                                            if heap_value_2 > heap_value_1:
                                                root_resident_1_path_first = 1
                                                root_resident_1_path_first_value = heap_value_1
                                                root_resident_1_path_first_index = heap_index_1
                                            if root_resident_1_path_first_value > incoming_value:
                                                heap_value_0 = incoming_value
                                                heap_index_0 = incoming_index
                                            else:
                                                heap_value_0 = root_resident_1_path_first_value
                                                heap_index_0 = root_resident_1_path_first_index
                                                root_resident_1_path_second = 6
                                                root_resident_1_path_second_value = heap_value_6
                                                root_resident_1_path_second_index = heap_index_6
                                                if root_resident_1_path_first == 1:
                                                    root_resident_1_path_second = 4
                                                    root_resident_1_path_second_value = heap_value_4
                                                    root_resident_1_path_second_index = heap_index_4
                                                    if heap_value_4 > heap_value_3:
                                                        root_resident_1_path_second = 3
                                                        root_resident_1_path_second_value = heap_value_3
                                                        root_resident_1_path_second_index = heap_index_3
                                                elif heap_value_6 > heap_value_5:
                                                    root_resident_1_path_second = 5
                                                    root_resident_1_path_second_value = heap_value_5
                                                    root_resident_1_path_second_index = heap_index_5
                                                if root_resident_1_path_second_value > incoming_value:
                                                    if root_resident_1_path_first == 1:
                                                        heap_value_1 = incoming_value
                                                        heap_index_1 = incoming_index
                                                    else:
                                                        heap_value_2 = incoming_value
                                                        heap_index_2 = incoming_index
                                                else:
                                                    if root_resident_1_path_first == 1:
                                                        heap_value_1 = root_resident_1_path_second_value
                                                        heap_index_1 = root_resident_1_path_second_index
                                                    else:
                                                        heap_value_2 = root_resident_1_path_second_value
                                                        heap_index_2 = root_resident_1_path_second_index
                                                    if root_resident_1_path_second == 3:
                                                        if heap_value_7 > incoming_value:
                                                            heap_value_3 = incoming_value
                                                            heap_index_3 = incoming_index
                                                        else:
                                                            heap_value_3 = heap_value_7
                                                            heap_index_3 = heap_index_7
                                                            heap_value_7 = incoming_value
                                                            heap_index_7 = incoming_index
                                                    elif root_resident_1_path_second == 4:
                                                        heap_value_4 = incoming_value
                                                        heap_index_4 = incoming_index
                                                    elif root_resident_1_path_second == 5:
                                                        heap_value_5 = incoming_value
                                                        heap_index_5 = incoming_index
                                                    else:
                                                        heap_value_6 = incoming_value
                                                        heap_index_6 = incoming_index
                    for block in pl.range((PARENT_SUMMARY_PREFIX_WIDTH + SUFFIX_SUMMARY_PREFIX_WIDTH) // RESIDENT_BLOCK_WIDTH, cols // RESIDENT_BLOCK_WIDTH):
                        column = block * RESIDENT_BLOCK_WIDTH
                        fallback_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                        block_max_lanes = pl.col_max(pl.reshape(fallback_probabilities, [RESIDENT_BLOCK_WIDTH // 8, 8]))
                        block_max = pl.read(block_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(block_max_lanes, [0, lane])
                            if lane_value > block_max:
                                block_max = lane_value
                        if block_max > heap_value_0:
                            leaf_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                            for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(leaf_probabilities, [0, block_offset])
                                    root_resident_2_path_first = 0
                                    root_resident_2_path_first_value = 0.0
                                    root_resident_2_path_first_index = pl.cast(0, pl.INT32)
                                    root_resident_2_path_second = 0
                                    root_resident_2_path_second_value = 0.0
                                    root_resident_2_path_second_index = pl.cast(0, pl.INT32)
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        root_resident_2_path_first = 2
                                        root_resident_2_path_first_value = heap_value_2
                                        root_resident_2_path_first_index = heap_index_2
                                        if heap_value_2 > heap_value_1:
                                            root_resident_2_path_first = 1
                                            root_resident_2_path_first_value = heap_value_1
                                            root_resident_2_path_first_index = heap_index_1
                                        if root_resident_2_path_first_value > incoming_value:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        else:
                                            heap_value_0 = root_resident_2_path_first_value
                                            heap_index_0 = root_resident_2_path_first_index
                                            root_resident_2_path_second = 6
                                            root_resident_2_path_second_value = heap_value_6
                                            root_resident_2_path_second_index = heap_index_6
                                            if root_resident_2_path_first == 1:
                                                root_resident_2_path_second = 4
                                                root_resident_2_path_second_value = heap_value_4
                                                root_resident_2_path_second_index = heap_index_4
                                                if heap_value_4 > heap_value_3:
                                                    root_resident_2_path_second = 3
                                                    root_resident_2_path_second_value = heap_value_3
                                                    root_resident_2_path_second_index = heap_index_3
                                            elif heap_value_6 > heap_value_5:
                                                root_resident_2_path_second = 5
                                                root_resident_2_path_second_value = heap_value_5
                                                root_resident_2_path_second_index = heap_index_5
                                            if root_resident_2_path_second_value > incoming_value:
                                                if root_resident_2_path_first == 1:
                                                    heap_value_1 = incoming_value
                                                    heap_index_1 = incoming_index
                                                else:
                                                    heap_value_2 = incoming_value
                                                    heap_index_2 = incoming_index
                                            else:
                                                if root_resident_2_path_first == 1:
                                                    heap_value_1 = root_resident_2_path_second_value
                                                    heap_index_1 = root_resident_2_path_second_index
                                                else:
                                                    heap_value_2 = root_resident_2_path_second_value
                                                    heap_index_2 = root_resident_2_path_second_index
                                                if root_resident_2_path_second == 3:
                                                    if heap_value_7 > incoming_value:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                    else:
                                                        heap_value_3 = heap_value_7
                                                        heap_index_3 = heap_index_7
                                                        heap_value_7 = incoming_value
                                                        heap_index_7 = incoming_index
                                                elif root_resident_2_path_second == 4:
                                                    heap_value_4 = incoming_value
                                                    heap_index_4 = incoming_index
                                                elif root_resident_2_path_second == 5:
                                                    heap_value_5 = incoming_value
                                                    heap_index_5 = incoming_index
                                                else:
                                                    heap_value_6 = incoming_value
                                                    heap_index_6 = incoming_index
                    if cols % RESIDENT_BLOCK_WIDTH != 0:
                        column = cols // RESIDENT_BLOCK_WIDTH * RESIDENT_BLOCK_WIDTH
                        block_probabilities = pl.slice(probabilities, [1, RESIDENT_TAIL_WIDTH], [0, column])
                        tail_max_lanes = pl.col_max(pl.reshape(block_probabilities, [RESIDENT_TAIL_WIDTH // 8, 8]))
                        tail_max = pl.read(tail_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(tail_max_lanes, [0, lane])
                            if lane_value > tail_max:
                                tail_max = lane_value
                        if tail_max > heap_value_0:
                            for block_offset in pl.range(RESIDENT_TAIL_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    root_resident_3_path_first = 0
                                    root_resident_3_path_first_value = 0.0
                                    root_resident_3_path_first_index = pl.cast(0, pl.INT32)
                                    root_resident_3_path_second = 0
                                    root_resident_3_path_second_value = 0.0
                                    root_resident_3_path_second_index = pl.cast(0, pl.INT32)
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        root_resident_3_path_first = 2
                                        root_resident_3_path_first_value = heap_value_2
                                        root_resident_3_path_first_index = heap_index_2
                                        if heap_value_2 > heap_value_1:
                                            root_resident_3_path_first = 1
                                            root_resident_3_path_first_value = heap_value_1
                                            root_resident_3_path_first_index = heap_index_1
                                        if root_resident_3_path_first_value > incoming_value:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        else:
                                            heap_value_0 = root_resident_3_path_first_value
                                            heap_index_0 = root_resident_3_path_first_index
                                            root_resident_3_path_second = 6
                                            root_resident_3_path_second_value = heap_value_6
                                            root_resident_3_path_second_index = heap_index_6
                                            if root_resident_3_path_first == 1:
                                                root_resident_3_path_second = 4
                                                root_resident_3_path_second_value = heap_value_4
                                                root_resident_3_path_second_index = heap_index_4
                                                if heap_value_4 > heap_value_3:
                                                    root_resident_3_path_second = 3
                                                    root_resident_3_path_second_value = heap_value_3
                                                    root_resident_3_path_second_index = heap_index_3
                                            elif heap_value_6 > heap_value_5:
                                                root_resident_3_path_second = 5
                                                root_resident_3_path_second_value = heap_value_5
                                                root_resident_3_path_second_index = heap_index_5
                                            if root_resident_3_path_second_value > incoming_value:
                                                if root_resident_3_path_first == 1:
                                                    heap_value_1 = incoming_value
                                                    heap_index_1 = incoming_index
                                                else:
                                                    heap_value_2 = incoming_value
                                                    heap_index_2 = incoming_index
                                            else:
                                                if root_resident_3_path_first == 1:
                                                    heap_value_1 = root_resident_3_path_second_value
                                                    heap_index_1 = root_resident_3_path_second_index
                                                else:
                                                    heap_value_2 = root_resident_3_path_second_value
                                                    heap_index_2 = root_resident_3_path_second_index
                                                if root_resident_3_path_second == 3:
                                                    if heap_value_7 > incoming_value:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                    else:
                                                        heap_value_3 = heap_value_7
                                                        heap_index_3 = heap_index_7
                                                        heap_value_7 = incoming_value
                                                        heap_index_7 = incoming_index
                                                elif root_resident_3_path_second == 4:
                                                    heap_value_4 = incoming_value
                                                    heap_index_4 = incoming_index
                                                elif root_resident_3_path_second == 5:
                                                    heap_value_5 = incoming_value
                                                    heap_index_5 = incoming_index
                                                else:
                                                    heap_value_6 = incoming_value
                                                    heap_index_6 = incoming_index
                    incoming_value = heap_value_7
                    incoming_index = heap_index_7
                    heap_value_7 = heap_value_0
                    heap_index_7 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    incoming_value = heap_value_6
                    incoming_index = heap_index_6
                    heap_value_6 = heap_value_0
                    heap_index_6 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            if heap_value_0 > incoming_value:
                                heap_value_2 = heap_value_0
                                heap_index_2 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_2 = incoming_value
                                heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    incoming_value = heap_value_5
                    incoming_index = heap_index_5
                    heap_value_5 = heap_value_0
                    heap_index_5 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_4
                    incoming_index = heap_index_4
                    heap_value_4 = heap_value_0
                    heap_index_4 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        if heap_value_1 > incoming_value:
                            heap_value_3 = heap_value_1
                            heap_index_3 = heap_index_1
                            if heap_value_0 > incoming_value:
                                heap_value_1 = heap_value_0
                                heap_index_1 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                        else:
                            heap_value_3 = incoming_value
                            heap_index_3 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_0
                    heap_index_3 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_0 > incoming_value:
                            heap_value_1 = heap_value_0
                            heap_index_1 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    heap_value_2 = heap_value_0
                    heap_index_2 = heap_index_0
                    heap_value_0 = heap_value_1
                    heap_index_0 = heap_index_1
                    if heap_value_0 > incoming_value:
                        heap_value_1 = heap_value_0
                        heap_index_1 = heap_index_0
                        heap_value_0 = incoming_value
                        heap_index_0 = incoming_index
                    else:
                        heap_value_1 = incoming_value
                        heap_index_1 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    heap_value_1 = heap_value_0
                    heap_index_1 = heap_index_0
                    heap_value_0 = incoming_value
                    heap_index_0 = incoming_index
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    pl.write(selected_values, [0, 0], heap_value_0)
                    pl.write(selected_indices, [0, 0], heap_index_0)
                    pl.write(selected_values, [0, 1], heap_value_1)
                    pl.write(selected_indices, [0, 1], heap_index_1)
                    pl.write(selected_values, [0, 2], heap_value_2)
                    pl.write(selected_indices, [0, 2], heap_index_2)
                    pl.write(selected_values, [0, 3], heap_value_3)
                    pl.write(selected_indices, [0, 3], heap_index_3)
                    pl.write(selected_values, [0, 4], heap_value_4)
                    pl.write(selected_indices, [0, 4], heap_index_4)
                    pl.write(selected_values, [0, 5], heap_value_5)
                    pl.write(selected_indices, [0, 5], heap_index_5)
                    pl.write(selected_values, [0, 6], heap_value_6)
                    pl.write(selected_indices, [0, 6], heap_index_6)
                    pl.write(selected_values, [0, 7], heap_value_7)
                    pl.write(selected_indices, [0, 7], heap_index_7)
                selected_sum = pl.read(selected_values, [0, 0])
                for slot in pl.unroll(1, 8):
                    selected_sum = selected_sum + pl.read(selected_values, [0, slot])
                weights = pl.div(selected_values, selected_sum + 1e-08)
                for slot in pl.unroll(8):
                    winner = pl.cast(pl.read(selected_indices, [0, slot]), pl.INDEX)
                    original_probability = pl.read(probabilities, [0, winner])
                    pl.write(probabilities, [0, winner], original_probability - 2.0)
                p = pl.assemble(p, probabilities, [row, 0])
                topk_w = pl.assemble(topk_w, weights, [row, 0])
                topk_idx = pl.assemble(topk_idx, selected_indices, [row, 0])
    return (topk_w, topk_idx, p)

@pl.jit
def resident_fp32(logits: pl.Tensor, colidx: pl.Tensor, p: pl.Out[pl.Tensor], topk_w: pl.Out[pl.Tensor], topk_idx: pl.Out[pl.Tensor]):
    rows, cols = logits.shape
    for row_block in pl.spmd((rows + ROWS_PER_TASK - 1) // ROWS_PER_TASK, name_hint='capacity_resident_gating'):
        row_base = row_block * ROWS_PER_TASK
        for row_offset in pl.range(ROWS_PER_TASK):
            row = row_base + row_offset
            if row < rows:
                input_values = pl.slice(logits, [1, cols], [row, 0])
                values = input_values
                maximum_lanes = pl.col_max(pl.reshape(values, [cols // 8, 8]))
                maximum = pl.read(maximum_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(maximum_lanes, [0, lane])
                    if lane_value > maximum:
                        maximum = lane_value
                exponentials = pl.exp(pl.sub(values, maximum))
                denominator_lanes = pl.col_sum(pl.reshape(exponentials, [cols // 8, 8]))
                denominator = pl.read(denominator_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(denominator_lanes, [0, lane])
                    denominator = denominator + lane_value
                probabilities = pl.div(exponentials, denominator)
                if cols < 512:
                    columns = pl.slice(colidx, [1, cols], [0, 0])
                    partition_values = pl.add(probabilities, 0.0)
                    partition_indices = pl.cast(columns, pl.INT32)
                    first = 0
                    last = cols
                    depth_limit = NARROW_DEPTH
                    for bounded_step_18 in pl.range(NARROW_DEPTH + 1):
                        if last - first > 3:
                            if depth_limit == 0:
                                heap_length = 8 - first
                                selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                                selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                                for slot in pl.range(heap_length):
                                    initial_value = pl.read(partition_values, [0, first + slot])
                                    initial_index = pl.read(partition_indices, [0, first + slot])
                                    pl.write(selected_values, [0, slot], initial_value)
                                    pl.write(selected_indices, [0, slot], initial_index)
                                if heap_length > 1:
                                    for parent_step in pl.range((heap_length - 2) // 2 + 1):
                                        hole = (heap_length - 2) // 2 - parent_step
                                        incoming_value = pl.read(selected_values, [0, hole])
                                        incoming_index = pl.read(selected_indices, [0, hole])
                                        hole_adjust_1 = hole
                                        top_adjust_1 = hole_adjust_1
                                        child_adjust_1 = hole_adjust_1
                                        for bounded_step_11 in pl.range(3):
                                            if child_adjust_1 < (heap_length - 1) // 2:
                                                child_adjust_1 = 2 * (child_adjust_1 + 1)
                                                right_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                left_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                                if right_value_adjust_1 > left_value_adjust_1:
                                                    child_adjust_1 = child_adjust_1 - 1
                                                moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1])
                                                pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                hole_adjust_1 = child_adjust_1
                                        if heap_length % 2 == 0 and child_adjust_1 == (heap_length - 2) // 2:
                                            child_adjust_1 = 2 * (child_adjust_1 + 1)
                                            moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                            moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1 - 1])
                                            pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                            pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                            hole_adjust_1 = child_adjust_1 - 1
                                        for bounded_step_12 in pl.range(3):
                                            if hole_adjust_1 > top_adjust_1:
                                                parent_adjust_1 = (hole_adjust_1 - 1) // 2
                                                parent_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                if not parent_value_adjust_1 <= incoming_value:
                                                    moving_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                    moving_index_adjust_1 = pl.read(selected_indices, [0, parent_adjust_1])
                                                    pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                    pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                    hole_adjust_1 = parent_adjust_1
                                        pl.write(selected_values, [0, hole_adjust_1], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_1], incoming_index)
                                for scan_column in pl.range(8, last):
                                    incoming_value = pl.read(partition_values, [0, scan_column])
                                    root = pl.read(selected_values, [0, 0])
                                    if incoming_value > root:
                                        incoming_index = pl.read(partition_indices, [0, scan_column])
                                        hole = 0
                                        hole_adjust_2 = hole
                                        top_adjust_2 = hole_adjust_2
                                        child_adjust_2 = hole_adjust_2
                                        for bounded_step_13 in pl.range(3):
                                            if child_adjust_2 < (heap_length - 1) // 2:
                                                child_adjust_2 = 2 * (child_adjust_2 + 1)
                                                right_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                left_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                                if right_value_adjust_2 > left_value_adjust_2:
                                                    child_adjust_2 = child_adjust_2 - 1
                                                moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2])
                                                pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                hole_adjust_2 = child_adjust_2
                                        if heap_length % 2 == 0 and child_adjust_2 == (heap_length - 2) // 2:
                                            child_adjust_2 = 2 * (child_adjust_2 + 1)
                                            moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                            moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2 - 1])
                                            pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                            pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                            hole_adjust_2 = child_adjust_2 - 1
                                        for bounded_step_14 in pl.range(3):
                                            if hole_adjust_2 > top_adjust_2:
                                                parent_adjust_2 = (hole_adjust_2 - 1) // 2
                                                parent_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                if not parent_value_adjust_2 <= incoming_value:
                                                    moving_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                    moving_index_adjust_2 = pl.read(selected_indices, [0, parent_adjust_2])
                                                    pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                    pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                    hole_adjust_2 = parent_adjust_2
                                        pl.write(selected_values, [0, hole_adjust_2], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_2], incoming_index)
                                for slot in pl.range(heap_length):
                                    copied_value = pl.read(selected_values, [0, slot])
                                    copied_index = pl.read(selected_indices, [0, slot])
                                    pl.write(partition_values, [0, first + slot], copied_value)
                                    pl.write(partition_indices, [0, first + slot], copied_index)
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, 7])
                                moving_index = pl.read(partition_indices, [0, 7])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, 7], swap_value)
                                pl.write(partition_indices, [0, 7], swap_index)
                                first = last
                            else:
                                depth_limit = depth_limit - 1
                                median_left = first + 1
                                median_middle = first + (last - first) // 2
                                median_right = last - 1
                                left_value = pl.read(partition_values, [0, median_left])
                                middle_value = pl.read(partition_values, [0, median_middle])
                                right_value = pl.read(partition_values, [0, median_right])
                                pivot_column = median_middle
                                if left_value > middle_value:
                                    if middle_value > right_value:
                                        pivot_column = median_middle
                                    elif left_value > right_value:
                                        pivot_column = median_right
                                    else:
                                        pivot_column = median_left
                                elif left_value > right_value:
                                    pivot_column = median_left
                                elif middle_value > right_value:
                                    pivot_column = median_right
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, pivot_column])
                                moving_index = pl.read(partition_indices, [0, pivot_column])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, pivot_column], swap_value)
                                pl.write(partition_indices, [0, pivot_column], swap_index)
                                pivot_value = pl.read(partition_values, [0, first])
                                partition_left = first + 1
                                partition_right = last
                                partition_active = True
                                for bounded_step_17 in pl.range(cols + 8):
                                    if partition_active:
                                        probe_left = pl.read(partition_values, [0, partition_left])
                                        for bounded_step_15 in pl.range(cols + 8):
                                            if probe_left > pivot_value:
                                                partition_left = partition_left + 1
                                                probe_left = pl.read(partition_values, [0, partition_left])
                                        partition_right = partition_right - 1
                                        probe_right = pl.read(partition_values, [0, partition_right])
                                        for bounded_step_16 in pl.range(cols + 8):
                                            if pivot_value > probe_right:
                                                partition_right = partition_right - 1
                                                probe_right = pl.read(partition_values, [0, partition_right])
                                        if partition_left >= partition_right:
                                            partition_active = False
                                        else:
                                            swap_value = pl.read(partition_values, [0, partition_left])
                                            swap_index = pl.read(partition_indices, [0, partition_left])
                                            moving_value = pl.read(partition_values, [0, partition_right])
                                            moving_index = pl.read(partition_indices, [0, partition_right])
                                            pl.write(partition_values, [0, partition_left], moving_value)
                                            pl.write(partition_indices, [0, partition_left], moving_index)
                                            pl.write(partition_values, [0, partition_right], swap_value)
                                            pl.write(partition_indices, [0, partition_right], swap_index)
                                            partition_left = partition_left + 1
                                if partition_left <= 7:
                                    first = partition_left
                                else:
                                    last = partition_left
                    for insertion_column in pl.range(first + 1, last):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_19 in pl.range(cols + 8):
                            if insertion_hole > first:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    for insertion_column in pl.range(0 + 1, 7):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_20 in pl.range(cols + 8):
                            if insertion_hole > 0:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    for slot in pl.range(8):
                        copied_value = pl.read(partition_values, [0, slot])
                        copied_index = pl.read(partition_indices, [0, slot])
                        pl.write(selected_values, [0, slot], copied_value)
                        pl.write(selected_indices, [0, slot], copied_index)
                else:
                    heap_value_0 = pl.read(probabilities, [0, 0])
                    heap_index_0 = pl.cast(0, pl.INT32)
                    heap_value_1 = pl.read(probabilities, [0, 1])
                    heap_index_1 = pl.cast(1, pl.INT32)
                    heap_value_2 = pl.read(probabilities, [0, 2])
                    heap_index_2 = pl.cast(2, pl.INT32)
                    heap_value_3 = pl.read(probabilities, [0, 3])
                    heap_index_3 = pl.cast(3, pl.INT32)
                    heap_value_4 = pl.read(probabilities, [0, 4])
                    heap_index_4 = pl.cast(4, pl.INT32)
                    heap_value_5 = pl.read(probabilities, [0, 5])
                    heap_index_5 = pl.cast(5, pl.INT32)
                    heap_value_6 = pl.read(probabilities, [0, 6])
                    heap_index_6 = pl.cast(6, pl.INT32)
                    heap_value_7 = pl.read(probabilities, [0, 7])
                    heap_index_7 = pl.cast(7, pl.INT32)
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_7
                    heap_index_3 = heap_index_7
                    if heap_value_3 > incoming_value:
                        heap_value_7 = heap_value_3
                        heap_index_7 = heap_index_3
                        heap_value_3 = incoming_value
                        heap_index_3 = incoming_index
                    else:
                        heap_value_7 = incoming_value
                        heap_index_7 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    if heap_value_6 > heap_value_5:
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    else:
                        heap_value_2 = heap_value_6
                        heap_index_2 = heap_index_6
                        if heap_value_2 > incoming_value:
                            heap_value_6 = heap_value_2
                            heap_index_6 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_6 = incoming_value
                            heap_index_6 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    if heap_value_4 > heap_value_3:
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        heap_value_3 = heap_value_7
                        heap_index_3 = heap_index_7
                        if heap_value_3 > incoming_value:
                            heap_value_7 = heap_value_3
                            heap_index_7 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_7 = incoming_value
                            heap_index_7 = incoming_index
                    else:
                        heap_value_1 = heap_value_4
                        heap_index_1 = heap_index_4
                        if heap_value_1 > incoming_value:
                            heap_value_4 = heap_value_1
                            heap_index_4 = heap_index_1
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                        else:
                            heap_value_4 = incoming_value
                            heap_index_4 = incoming_index
                    incoming_value = heap_value_0
                    incoming_index = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            heap_value_3 = heap_value_7
                            heap_index_3 = heap_index_7
                            if heap_value_3 > incoming_value:
                                heap_value_7 = heap_value_3
                                heap_index_7 = heap_index_3
                                if heap_value_1 > incoming_value:
                                    heap_value_3 = heap_value_1
                                    heap_index_3 = heap_index_1
                                    if heap_value_0 > incoming_value:
                                        heap_value_1 = heap_value_0
                                        heap_index_1 = heap_index_0
                                        heap_value_0 = incoming_value
                                        heap_index_0 = incoming_index
                                    else:
                                        heap_value_1 = incoming_value
                                        heap_index_1 = incoming_index
                                else:
                                    heap_value_3 = incoming_value
                                    heap_index_3 = incoming_index
                            else:
                                heap_value_7 = incoming_value
                                heap_index_7 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    path_active = False
                    path_hole = 0
                    path_second = 0
                    path_second_value = heap_value_0
                    path_second_index = heap_index_0
                    block_max = heap_value_0
                    if PARENT_SUMMARY_PREFIX_WIDTH > 0:
                        parent_prefix = pl.slice(probabilities, [1, PARENT_SUMMARY_PREFIX_WIDTH], [0, 0])
                        parent_values = pl.reshape(parent_prefix, [PARENT_SUMMARY_PREFIX_WIDTH // PARENT_WIDTH, PARENT_WIDTH])
                        parent_maxima = pl.row_max(parent_values)
                        for parent in pl.range(PARENT_SUMMARY_PREFIX_WIDTH // PARENT_WIDTH):
                            parent_max = pl.read(parent_maxima, [parent, 0])
                            if parent_max > heap_value_0:
                                parent_probabilities = pl.slice(probabilities, [1, PARENT_WIDTH], [0, parent * PARENT_WIDTH])
                                child_values = pl.reshape(parent_probabilities, [PARENT_CHILDREN, RESIDENT_BLOCK_WIDTH])
                                child_maxima = pl.row_max(child_values)
                                for child in pl.range(PARENT_CHILDREN):
                                    block = parent * PARENT_CHILDREN + child
                                    column = block * RESIDENT_BLOCK_WIDTH
                                    block_max = pl.read(child_maxima, [child, 0])
                                    if block_max > heap_value_0:
                                        leaf_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                                        for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                            candidate_column = column + block_offset
                                            if candidate_column >= 8:
                                                incoming_value = pl.read(leaf_probabilities, [0, block_offset])
                                                if incoming_value > heap_value_0:
                                                    incoming_index = pl.cast(candidate_column, pl.INT32)
                                                    path_first = 2
                                                    path_first_value = heap_value_2
                                                    path_first_index = heap_index_2
                                                    path_second = 6
                                                    path_second_value = heap_value_6
                                                    path_second_index = heap_index_6
                                                    if heap_value_2 > heap_value_1:
                                                        path_first = 1
                                                        path_first_value = heap_value_1
                                                        path_first_index = heap_index_1
                                                        if heap_value_4 > heap_value_3:
                                                            path_second = 3
                                                            path_second_value = heap_value_3
                                                            path_second_index = heap_index_3
                                                        else:
                                                            path_second = 4
                                                            path_second_value = heap_value_4
                                                            path_second_index = heap_index_4
                                                    else:
                                                        path_first = 2
                                                        path_first_value = heap_value_2
                                                        path_first_index = heap_index_2
                                                        if heap_value_6 > heap_value_5:
                                                            path_second = 5
                                                            path_second_value = heap_value_5
                                                            path_second_index = heap_index_5
                                                        else:
                                                            path_second = 6
                                                            path_second_value = heap_value_6
                                                            path_second_index = heap_index_6
                                                    heap_value_0 = path_first_value
                                                    heap_index_0 = path_first_index
                                                    if path_first == 1:
                                                        heap_value_1 = path_second_value
                                                        heap_index_1 = path_second_index
                                                    else:
                                                        heap_value_2 = path_second_value
                                                        heap_index_2 = path_second_index
                                                    path_hole = path_second
                                                    path_active = True
                                                    if path_second == 3:
                                                        path_third_value = heap_value_7
                                                        path_third_index = heap_index_7
                                                        heap_value_3 = path_third_value
                                                        heap_index_3 = path_third_index
                                                        path_hole = 7
                                                        if path_third_value > incoming_value:
                                                            heap_value_7 = path_third_value
                                                            heap_index_7 = path_third_index
                                                            path_hole = 3
                                                        else:
                                                            path_active = False
                                                    if path_active:
                                                        if path_second_value > incoming_value:
                                                            if path_hole == 3:
                                                                heap_value_3 = path_second_value
                                                                heap_index_3 = path_second_index
                                                            elif path_hole == 4:
                                                                heap_value_4 = path_second_value
                                                                heap_index_4 = path_second_index
                                                            elif path_hole == 5:
                                                                heap_value_5 = path_second_value
                                                                heap_index_5 = path_second_index
                                                            elif path_hole == 6:
                                                                heap_value_6 = path_second_value
                                                                heap_index_6 = path_second_index
                                                            path_hole = path_first
                                                        else:
                                                            path_active = False
                                                    if path_active:
                                                        if path_first_value > incoming_value:
                                                            if path_hole == 1:
                                                                heap_value_1 = path_first_value
                                                                heap_index_1 = path_first_index
                                                            elif path_hole == 2:
                                                                heap_value_2 = path_first_value
                                                                heap_index_2 = path_first_index
                                                            path_hole = 0
                                                    if path_hole == 0:
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    elif path_hole == 1:
                                                        heap_value_1 = incoming_value
                                                        heap_index_1 = incoming_index
                                                    elif path_hole == 2:
                                                        heap_value_2 = incoming_value
                                                        heap_index_2 = incoming_index
                                                    elif path_hole == 3:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                    elif path_hole == 4:
                                                        heap_value_4 = incoming_value
                                                        heap_index_4 = incoming_index
                                                    elif path_hole == 5:
                                                        heap_value_5 = incoming_value
                                                        heap_index_5 = incoming_index
                                                    elif path_hole == 6:
                                                        heap_value_6 = incoming_value
                                                        heap_index_6 = incoming_index
                                                    elif path_hole == 7:
                                                        heap_value_7 = incoming_value
                                                        heap_index_7 = incoming_index
                    if SUFFIX_SUMMARY_PREFIX_WIDTH > 0:
                        suffix_prefix = pl.slice(probabilities, [1, SUFFIX_SUMMARY_TILE_WIDTH], [0, SUFFIX_SUMMARY_OFFSET])
                        suffix_values = pl.reshape(suffix_prefix, [SUFFIX_SUMMARY_TILE_WIDTH // RESIDENT_BLOCK_WIDTH, RESIDENT_BLOCK_WIDTH])
                        suffix_maxima = pl.row_max(suffix_values)
                        for block in pl.range(PARENT_SUMMARY_PREFIX_WIDTH // RESIDENT_BLOCK_WIDTH, (PARENT_SUMMARY_PREFIX_WIDTH + SUFFIX_SUMMARY_PREFIX_WIDTH) // RESIDENT_BLOCK_WIDTH):
                            column = block * RESIDENT_BLOCK_WIDTH
                            block_max = pl.read(suffix_maxima, [block - PARENT_SUMMARY_PREFIX_WIDTH // RESIDENT_BLOCK_WIDTH, 0])
                            if block_max > heap_value_0:
                                leaf_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                                for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                    candidate_column = column + block_offset
                                    if candidate_column >= 8:
                                        incoming_value = pl.read(leaf_probabilities, [0, block_offset])
                                        if incoming_value > heap_value_0:
                                            incoming_index = pl.cast(candidate_column, pl.INT32)
                                            path_first = 2
                                            path_first_value = heap_value_2
                                            path_first_index = heap_index_2
                                            path_second = 6
                                            path_second_value = heap_value_6
                                            path_second_index = heap_index_6
                                            if heap_value_2 > heap_value_1:
                                                path_first = 1
                                                path_first_value = heap_value_1
                                                path_first_index = heap_index_1
                                                if heap_value_4 > heap_value_3:
                                                    path_second = 3
                                                    path_second_value = heap_value_3
                                                    path_second_index = heap_index_3
                                                else:
                                                    path_second = 4
                                                    path_second_value = heap_value_4
                                                    path_second_index = heap_index_4
                                            else:
                                                path_first = 2
                                                path_first_value = heap_value_2
                                                path_first_index = heap_index_2
                                                if heap_value_6 > heap_value_5:
                                                    path_second = 5
                                                    path_second_value = heap_value_5
                                                    path_second_index = heap_index_5
                                                else:
                                                    path_second = 6
                                                    path_second_value = heap_value_6
                                                    path_second_index = heap_index_6
                                            heap_value_0 = path_first_value
                                            heap_index_0 = path_first_index
                                            if path_first == 1:
                                                heap_value_1 = path_second_value
                                                heap_index_1 = path_second_index
                                            else:
                                                heap_value_2 = path_second_value
                                                heap_index_2 = path_second_index
                                            path_hole = path_second
                                            path_active = True
                                            if path_second == 3:
                                                path_third_value = heap_value_7
                                                path_third_index = heap_index_7
                                                heap_value_3 = path_third_value
                                                heap_index_3 = path_third_index
                                                path_hole = 7
                                                if path_third_value > incoming_value:
                                                    heap_value_7 = path_third_value
                                                    heap_index_7 = path_third_index
                                                    path_hole = 3
                                                else:
                                                    path_active = False
                                            if path_active:
                                                if path_second_value > incoming_value:
                                                    if path_hole == 3:
                                                        heap_value_3 = path_second_value
                                                        heap_index_3 = path_second_index
                                                    elif path_hole == 4:
                                                        heap_value_4 = path_second_value
                                                        heap_index_4 = path_second_index
                                                    elif path_hole == 5:
                                                        heap_value_5 = path_second_value
                                                        heap_index_5 = path_second_index
                                                    elif path_hole == 6:
                                                        heap_value_6 = path_second_value
                                                        heap_index_6 = path_second_index
                                                    path_hole = path_first
                                                else:
                                                    path_active = False
                                            if path_active:
                                                if path_first_value > incoming_value:
                                                    if path_hole == 1:
                                                        heap_value_1 = path_first_value
                                                        heap_index_1 = path_first_index
                                                    elif path_hole == 2:
                                                        heap_value_2 = path_first_value
                                                        heap_index_2 = path_first_index
                                                    path_hole = 0
                                            if path_hole == 0:
                                                heap_value_0 = incoming_value
                                                heap_index_0 = incoming_index
                                            elif path_hole == 1:
                                                heap_value_1 = incoming_value
                                                heap_index_1 = incoming_index
                                            elif path_hole == 2:
                                                heap_value_2 = incoming_value
                                                heap_index_2 = incoming_index
                                            elif path_hole == 3:
                                                heap_value_3 = incoming_value
                                                heap_index_3 = incoming_index
                                            elif path_hole == 4:
                                                heap_value_4 = incoming_value
                                                heap_index_4 = incoming_index
                                            elif path_hole == 5:
                                                heap_value_5 = incoming_value
                                                heap_index_5 = incoming_index
                                            elif path_hole == 6:
                                                heap_value_6 = incoming_value
                                                heap_index_6 = incoming_index
                                            elif path_hole == 7:
                                                heap_value_7 = incoming_value
                                                heap_index_7 = incoming_index
                    for block in pl.range((PARENT_SUMMARY_PREFIX_WIDTH + SUFFIX_SUMMARY_PREFIX_WIDTH) // RESIDENT_BLOCK_WIDTH, cols // RESIDENT_BLOCK_WIDTH):
                        column = block * RESIDENT_BLOCK_WIDTH
                        fallback_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                        block_max_lanes = pl.col_max(pl.reshape(fallback_probabilities, [RESIDENT_BLOCK_WIDTH // 8, 8]))
                        block_max = pl.read(block_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(block_max_lanes, [0, lane])
                            if lane_value > block_max:
                                block_max = lane_value
                        if block_max > heap_value_0:
                            leaf_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                            for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(leaf_probabilities, [0, block_offset])
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        path_first = 2
                                        path_first_value = heap_value_2
                                        path_first_index = heap_index_2
                                        path_second = 6
                                        path_second_value = heap_value_6
                                        path_second_index = heap_index_6
                                        if heap_value_2 > heap_value_1:
                                            path_first = 1
                                            path_first_value = heap_value_1
                                            path_first_index = heap_index_1
                                            if heap_value_4 > heap_value_3:
                                                path_second = 3
                                                path_second_value = heap_value_3
                                                path_second_index = heap_index_3
                                            else:
                                                path_second = 4
                                                path_second_value = heap_value_4
                                                path_second_index = heap_index_4
                                        else:
                                            path_first = 2
                                            path_first_value = heap_value_2
                                            path_first_index = heap_index_2
                                            if heap_value_6 > heap_value_5:
                                                path_second = 5
                                                path_second_value = heap_value_5
                                                path_second_index = heap_index_5
                                            else:
                                                path_second = 6
                                                path_second_value = heap_value_6
                                                path_second_index = heap_index_6
                                        heap_value_0 = path_first_value
                                        heap_index_0 = path_first_index
                                        if path_first == 1:
                                            heap_value_1 = path_second_value
                                            heap_index_1 = path_second_index
                                        else:
                                            heap_value_2 = path_second_value
                                            heap_index_2 = path_second_index
                                        path_hole = path_second
                                        path_active = True
                                        if path_second == 3:
                                            path_third_value = heap_value_7
                                            path_third_index = heap_index_7
                                            heap_value_3 = path_third_value
                                            heap_index_3 = path_third_index
                                            path_hole = 7
                                            if path_third_value > incoming_value:
                                                heap_value_7 = path_third_value
                                                heap_index_7 = path_third_index
                                                path_hole = 3
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_second_value > incoming_value:
                                                if path_hole == 3:
                                                    heap_value_3 = path_second_value
                                                    heap_index_3 = path_second_index
                                                elif path_hole == 4:
                                                    heap_value_4 = path_second_value
                                                    heap_index_4 = path_second_index
                                                elif path_hole == 5:
                                                    heap_value_5 = path_second_value
                                                    heap_index_5 = path_second_index
                                                elif path_hole == 6:
                                                    heap_value_6 = path_second_value
                                                    heap_index_6 = path_second_index
                                                path_hole = path_first
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_first_value > incoming_value:
                                                if path_hole == 1:
                                                    heap_value_1 = path_first_value
                                                    heap_index_1 = path_first_index
                                                elif path_hole == 2:
                                                    heap_value_2 = path_first_value
                                                    heap_index_2 = path_first_index
                                                path_hole = 0
                                        if path_hole == 0:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        elif path_hole == 1:
                                            heap_value_1 = incoming_value
                                            heap_index_1 = incoming_index
                                        elif path_hole == 2:
                                            heap_value_2 = incoming_value
                                            heap_index_2 = incoming_index
                                        elif path_hole == 3:
                                            heap_value_3 = incoming_value
                                            heap_index_3 = incoming_index
                                        elif path_hole == 4:
                                            heap_value_4 = incoming_value
                                            heap_index_4 = incoming_index
                                        elif path_hole == 5:
                                            heap_value_5 = incoming_value
                                            heap_index_5 = incoming_index
                                        elif path_hole == 6:
                                            heap_value_6 = incoming_value
                                            heap_index_6 = incoming_index
                                        elif path_hole == 7:
                                            heap_value_7 = incoming_value
                                            heap_index_7 = incoming_index
                    if cols % RESIDENT_BLOCK_WIDTH != 0:
                        column = cols // RESIDENT_BLOCK_WIDTH * RESIDENT_BLOCK_WIDTH
                        block_probabilities = pl.slice(probabilities, [1, RESIDENT_TAIL_WIDTH], [0, column])
                        tail_max_lanes = pl.col_max(pl.reshape(block_probabilities, [RESIDENT_TAIL_WIDTH // 8, 8]))
                        tail_max = pl.read(tail_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(tail_max_lanes, [0, lane])
                            if lane_value > tail_max:
                                tail_max = lane_value
                        if tail_max > heap_value_0:
                            for block_offset in pl.range(RESIDENT_TAIL_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        path_first = 2
                                        path_first_value = heap_value_2
                                        path_first_index = heap_index_2
                                        path_second = 6
                                        path_second_value = heap_value_6
                                        path_second_index = heap_index_6
                                        if heap_value_2 > heap_value_1:
                                            path_first = 1
                                            path_first_value = heap_value_1
                                            path_first_index = heap_index_1
                                            if heap_value_4 > heap_value_3:
                                                path_second = 3
                                                path_second_value = heap_value_3
                                                path_second_index = heap_index_3
                                            else:
                                                path_second = 4
                                                path_second_value = heap_value_4
                                                path_second_index = heap_index_4
                                        else:
                                            path_first = 2
                                            path_first_value = heap_value_2
                                            path_first_index = heap_index_2
                                            if heap_value_6 > heap_value_5:
                                                path_second = 5
                                                path_second_value = heap_value_5
                                                path_second_index = heap_index_5
                                            else:
                                                path_second = 6
                                                path_second_value = heap_value_6
                                                path_second_index = heap_index_6
                                        heap_value_0 = path_first_value
                                        heap_index_0 = path_first_index
                                        if path_first == 1:
                                            heap_value_1 = path_second_value
                                            heap_index_1 = path_second_index
                                        else:
                                            heap_value_2 = path_second_value
                                            heap_index_2 = path_second_index
                                        path_hole = path_second
                                        path_active = True
                                        if path_second == 3:
                                            path_third_value = heap_value_7
                                            path_third_index = heap_index_7
                                            heap_value_3 = path_third_value
                                            heap_index_3 = path_third_index
                                            path_hole = 7
                                            if path_third_value > incoming_value:
                                                heap_value_7 = path_third_value
                                                heap_index_7 = path_third_index
                                                path_hole = 3
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_second_value > incoming_value:
                                                if path_hole == 3:
                                                    heap_value_3 = path_second_value
                                                    heap_index_3 = path_second_index
                                                elif path_hole == 4:
                                                    heap_value_4 = path_second_value
                                                    heap_index_4 = path_second_index
                                                elif path_hole == 5:
                                                    heap_value_5 = path_second_value
                                                    heap_index_5 = path_second_index
                                                elif path_hole == 6:
                                                    heap_value_6 = path_second_value
                                                    heap_index_6 = path_second_index
                                                path_hole = path_first
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_first_value > incoming_value:
                                                if path_hole == 1:
                                                    heap_value_1 = path_first_value
                                                    heap_index_1 = path_first_index
                                                elif path_hole == 2:
                                                    heap_value_2 = path_first_value
                                                    heap_index_2 = path_first_index
                                                path_hole = 0
                                        if path_hole == 0:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        elif path_hole == 1:
                                            heap_value_1 = incoming_value
                                            heap_index_1 = incoming_index
                                        elif path_hole == 2:
                                            heap_value_2 = incoming_value
                                            heap_index_2 = incoming_index
                                        elif path_hole == 3:
                                            heap_value_3 = incoming_value
                                            heap_index_3 = incoming_index
                                        elif path_hole == 4:
                                            heap_value_4 = incoming_value
                                            heap_index_4 = incoming_index
                                        elif path_hole == 5:
                                            heap_value_5 = incoming_value
                                            heap_index_5 = incoming_index
                                        elif path_hole == 6:
                                            heap_value_6 = incoming_value
                                            heap_index_6 = incoming_index
                                        elif path_hole == 7:
                                            heap_value_7 = incoming_value
                                            heap_index_7 = incoming_index
                    incoming_value = heap_value_7
                    incoming_index = heap_index_7
                    heap_value_7 = heap_value_0
                    heap_index_7 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    incoming_value = heap_value_6
                    incoming_index = heap_index_6
                    heap_value_6 = heap_value_0
                    heap_index_6 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            if heap_value_0 > incoming_value:
                                heap_value_2 = heap_value_0
                                heap_index_2 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_2 = incoming_value
                                heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    incoming_value = heap_value_5
                    incoming_index = heap_index_5
                    heap_value_5 = heap_value_0
                    heap_index_5 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_4
                    incoming_index = heap_index_4
                    heap_value_4 = heap_value_0
                    heap_index_4 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        if heap_value_1 > incoming_value:
                            heap_value_3 = heap_value_1
                            heap_index_3 = heap_index_1
                            if heap_value_0 > incoming_value:
                                heap_value_1 = heap_value_0
                                heap_index_1 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                        else:
                            heap_value_3 = incoming_value
                            heap_index_3 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_0
                    heap_index_3 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_0 > incoming_value:
                            heap_value_1 = heap_value_0
                            heap_index_1 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    heap_value_2 = heap_value_0
                    heap_index_2 = heap_index_0
                    heap_value_0 = heap_value_1
                    heap_index_0 = heap_index_1
                    if heap_value_0 > incoming_value:
                        heap_value_1 = heap_value_0
                        heap_index_1 = heap_index_0
                        heap_value_0 = incoming_value
                        heap_index_0 = incoming_index
                    else:
                        heap_value_1 = incoming_value
                        heap_index_1 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    heap_value_1 = heap_value_0
                    heap_index_1 = heap_index_0
                    heap_value_0 = incoming_value
                    heap_index_0 = incoming_index
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    pl.write(selected_values, [0, 0], heap_value_0)
                    pl.write(selected_indices, [0, 0], heap_index_0)
                    pl.write(selected_values, [0, 1], heap_value_1)
                    pl.write(selected_indices, [0, 1], heap_index_1)
                    pl.write(selected_values, [0, 2], heap_value_2)
                    pl.write(selected_indices, [0, 2], heap_index_2)
                    pl.write(selected_values, [0, 3], heap_value_3)
                    pl.write(selected_indices, [0, 3], heap_index_3)
                    pl.write(selected_values, [0, 4], heap_value_4)
                    pl.write(selected_indices, [0, 4], heap_index_4)
                    pl.write(selected_values, [0, 5], heap_value_5)
                    pl.write(selected_indices, [0, 5], heap_index_5)
                    pl.write(selected_values, [0, 6], heap_value_6)
                    pl.write(selected_indices, [0, 6], heap_index_6)
                    pl.write(selected_values, [0, 7], heap_value_7)
                    pl.write(selected_indices, [0, 7], heap_index_7)
                selected_sum = pl.read(selected_values, [0, 0])
                for slot in pl.unroll(1, 8):
                    selected_sum = selected_sum + pl.read(selected_values, [0, slot])
                weights = pl.div(selected_values, selected_sum + 1e-08)
                for slot in pl.unroll(8):
                    winner = pl.cast(pl.read(selected_indices, [0, slot]), pl.INDEX)
                    original_probability = pl.read(probabilities, [0, winner])
                    pl.write(probabilities, [0, winner], original_probability - 2.0)
                p = pl.assemble(p, probabilities, [row, 0])
                topk_w = pl.assemble(topk_w, weights, [row, 0])
                topk_idx = pl.assemble(topk_idx, selected_indices, [row, 0])
    return (topk_w, topk_idx, p)

@pl.jit
def bounded(logits: pl.Tensor, colidx: pl.Tensor, p: pl.Out[pl.Tensor], topk_w: pl.Out[pl.Tensor], topk_idx: pl.Out[pl.Tensor]):
    rows, cols = logits.shape
    for row_base in pl.parallel(0, rows, ROWS_PER_TASK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='capacity_bounded_gating'):
            for row_offset in pl.range(ROWS_PER_TASK):
                row = row_base + row_offset
                if row < rows:
                    maximum_input = pl.slice(logits, [1, CHUNK_WIDTH], [row, 0])
                    maximum_values = pl.cast(maximum_input, pl.FP32)
                    maximum_lanes = pl.col_max(pl.reshape(maximum_values, [CHUNK_WIDTH // 8, 8]))
                    maximum = pl.read(maximum_lanes, [0, 0])
                    for lane in pl.unroll(1, 8):
                        lane_value = pl.read(maximum_lanes, [0, lane])
                        if lane_value > maximum:
                            maximum = lane_value
                    for chunk in pl.range(1, cols // CHUNK_WIDTH):
                        column = chunk * CHUNK_WIDTH
                        input_values = pl.slice(logits, [1, CHUNK_WIDTH], [row, column])
                        values = pl.cast(input_values, pl.FP32)
                        chunk_max_lanes = pl.col_max(pl.reshape(values, [CHUNK_WIDTH // 8, 8]))
                        chunk_max = pl.read(chunk_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(chunk_max_lanes, [0, lane])
                            if lane_value > chunk_max:
                                chunk_max = lane_value
                        if chunk_max > maximum:
                            maximum = chunk_max
                    denominator = 0.0
                    for chunk in pl.range(cols // CHUNK_WIDTH):
                        column = chunk * CHUNK_WIDTH
                        input_values = pl.slice(logits, [1, CHUNK_WIDTH], [row, column])
                        values = pl.cast(input_values, pl.FP32)
                        exponentials = pl.exp(pl.sub(values, maximum))
                        chunk_sum_lanes = pl.col_sum(pl.reshape(exponentials, [CHUNK_WIDTH // 8, 8]))
                        chunk_sum = pl.read(chunk_sum_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(chunk_sum_lanes, [0, lane])
                            chunk_sum = chunk_sum + lane_value
                        denominator = denominator + chunk_sum
                    first_input = pl.slice(logits, [1, CHUNK_WIDTH], [row, 0])
                    first_values = pl.cast(first_input, pl.FP32)
                    probabilities = pl.div(pl.exp(pl.sub(first_values, maximum)), denominator)
                    heap_value_0 = pl.read(probabilities, [0, 0])
                    heap_index_0 = pl.cast(0, pl.INT32)
                    heap_value_1 = pl.read(probabilities, [0, 1])
                    heap_index_1 = pl.cast(1, pl.INT32)
                    heap_value_2 = pl.read(probabilities, [0, 2])
                    heap_index_2 = pl.cast(2, pl.INT32)
                    heap_value_3 = pl.read(probabilities, [0, 3])
                    heap_index_3 = pl.cast(3, pl.INT32)
                    heap_value_4 = pl.read(probabilities, [0, 4])
                    heap_index_4 = pl.cast(4, pl.INT32)
                    heap_value_5 = pl.read(probabilities, [0, 5])
                    heap_index_5 = pl.cast(5, pl.INT32)
                    heap_value_6 = pl.read(probabilities, [0, 6])
                    heap_index_6 = pl.cast(6, pl.INT32)
                    heap_value_7 = pl.read(probabilities, [0, 7])
                    heap_index_7 = pl.cast(7, pl.INT32)
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_7
                    heap_index_3 = heap_index_7
                    if heap_value_3 > incoming_value:
                        heap_value_7 = heap_value_3
                        heap_index_7 = heap_index_3
                        heap_value_3 = incoming_value
                        heap_index_3 = incoming_index
                    else:
                        heap_value_7 = incoming_value
                        heap_index_7 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    if heap_value_6 > heap_value_5:
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    else:
                        heap_value_2 = heap_value_6
                        heap_index_2 = heap_index_6
                        if heap_value_2 > incoming_value:
                            heap_value_6 = heap_value_2
                            heap_index_6 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_6 = incoming_value
                            heap_index_6 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    if heap_value_4 > heap_value_3:
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        heap_value_3 = heap_value_7
                        heap_index_3 = heap_index_7
                        if heap_value_3 > incoming_value:
                            heap_value_7 = heap_value_3
                            heap_index_7 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_7 = incoming_value
                            heap_index_7 = incoming_index
                    else:
                        heap_value_1 = heap_value_4
                        heap_index_1 = heap_index_4
                        if heap_value_1 > incoming_value:
                            heap_value_4 = heap_value_1
                            heap_index_4 = heap_index_1
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                        else:
                            heap_value_4 = incoming_value
                            heap_index_4 = incoming_index
                    incoming_value = heap_value_0
                    incoming_index = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            heap_value_3 = heap_value_7
                            heap_index_3 = heap_index_7
                            if heap_value_3 > incoming_value:
                                heap_value_7 = heap_value_3
                                heap_index_7 = heap_index_3
                                if heap_value_1 > incoming_value:
                                    heap_value_3 = heap_value_1
                                    heap_index_3 = heap_index_1
                                    if heap_value_0 > incoming_value:
                                        heap_value_1 = heap_value_0
                                        heap_index_1 = heap_index_0
                                        heap_value_0 = incoming_value
                                        heap_index_0 = incoming_index
                                    else:
                                        heap_value_1 = incoming_value
                                        heap_index_1 = incoming_index
                                else:
                                    heap_value_3 = incoming_value
                                    heap_index_3 = incoming_index
                            else:
                                heap_value_7 = incoming_value
                                heap_index_7 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    for block in pl.range(cols // BLOCK_WIDTH):
                        column = block * BLOCK_WIDTH
                        selection_input = pl.slice(logits, [1, BLOCK_WIDTH], [row, column])
                        selection_values = pl.cast(selection_input, pl.FP32)
                        block_probabilities = pl.div(pl.exp(pl.sub(selection_values, maximum)), denominator)
                        block_max_lanes = pl.col_max(pl.reshape(block_probabilities, [BLOCK_WIDTH // 8, 8]))
                        block_max = pl.read(block_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(block_max_lanes, [0, lane])
                            if lane_value > block_max:
                                block_max = lane_value
                        if block_max > heap_value_0:
                            for block_offset in pl.range(BLOCK_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        if heap_value_2 > heap_value_1:
                                            heap_value_0 = heap_value_1
                                            heap_index_0 = heap_index_1
                                            if heap_value_4 > heap_value_3:
                                                heap_value_1 = heap_value_3
                                                heap_index_1 = heap_index_3
                                                heap_value_3 = heap_value_7
                                                heap_index_3 = heap_index_7
                                                if heap_value_3 > incoming_value:
                                                    heap_value_7 = heap_value_3
                                                    heap_index_7 = heap_index_3
                                                    if heap_value_1 > incoming_value:
                                                        heap_value_3 = heap_value_1
                                                        heap_index_3 = heap_index_1
                                                        if heap_value_0 > incoming_value:
                                                            heap_value_1 = heap_value_0
                                                            heap_index_1 = heap_index_0
                                                            heap_value_0 = incoming_value
                                                            heap_index_0 = incoming_index
                                                        else:
                                                            heap_value_1 = incoming_value
                                                            heap_index_1 = incoming_index
                                                    else:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                else:
                                                    heap_value_7 = incoming_value
                                                    heap_index_7 = incoming_index
                                            else:
                                                heap_value_1 = heap_value_4
                                                heap_index_1 = heap_index_4
                                                if heap_value_1 > incoming_value:
                                                    heap_value_4 = heap_value_1
                                                    heap_index_4 = heap_index_1
                                                    if heap_value_0 > incoming_value:
                                                        heap_value_1 = heap_value_0
                                                        heap_index_1 = heap_index_0
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    else:
                                                        heap_value_1 = incoming_value
                                                        heap_index_1 = incoming_index
                                                else:
                                                    heap_value_4 = incoming_value
                                                    heap_index_4 = incoming_index
                                        else:
                                            heap_value_0 = heap_value_2
                                            heap_index_0 = heap_index_2
                                            if heap_value_6 > heap_value_5:
                                                heap_value_2 = heap_value_5
                                                heap_index_2 = heap_index_5
                                                if heap_value_2 > incoming_value:
                                                    heap_value_5 = heap_value_2
                                                    heap_index_5 = heap_index_2
                                                    if heap_value_0 > incoming_value:
                                                        heap_value_2 = heap_value_0
                                                        heap_index_2 = heap_index_0
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    else:
                                                        heap_value_2 = incoming_value
                                                        heap_index_2 = incoming_index
                                                else:
                                                    heap_value_5 = incoming_value
                                                    heap_index_5 = incoming_index
                                            else:
                                                heap_value_2 = heap_value_6
                                                heap_index_2 = heap_index_6
                                                if heap_value_2 > incoming_value:
                                                    heap_value_6 = heap_value_2
                                                    heap_index_6 = heap_index_2
                                                    if heap_value_0 > incoming_value:
                                                        heap_value_2 = heap_value_0
                                                        heap_index_2 = heap_index_0
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    else:
                                                        heap_value_2 = incoming_value
                                                        heap_index_2 = incoming_index
                                                else:
                                                    heap_value_6 = incoming_value
                                                    heap_index_6 = incoming_index
                    incoming_value = heap_value_7
                    incoming_index = heap_index_7
                    heap_value_7 = heap_value_0
                    heap_index_7 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    incoming_value = heap_value_6
                    incoming_index = heap_index_6
                    heap_value_6 = heap_value_0
                    heap_index_6 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            if heap_value_0 > incoming_value:
                                heap_value_2 = heap_value_0
                                heap_index_2 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_2 = incoming_value
                                heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    incoming_value = heap_value_5
                    incoming_index = heap_index_5
                    heap_value_5 = heap_value_0
                    heap_index_5 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_4
                    incoming_index = heap_index_4
                    heap_value_4 = heap_value_0
                    heap_index_4 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        if heap_value_1 > incoming_value:
                            heap_value_3 = heap_value_1
                            heap_index_3 = heap_index_1
                            if heap_value_0 > incoming_value:
                                heap_value_1 = heap_value_0
                                heap_index_1 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                        else:
                            heap_value_3 = incoming_value
                            heap_index_3 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_0
                    heap_index_3 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_0 > incoming_value:
                            heap_value_1 = heap_value_0
                            heap_index_1 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    heap_value_2 = heap_value_0
                    heap_index_2 = heap_index_0
                    heap_value_0 = heap_value_1
                    heap_index_0 = heap_index_1
                    if heap_value_0 > incoming_value:
                        heap_value_1 = heap_value_0
                        heap_index_1 = heap_index_0
                        heap_value_0 = incoming_value
                        heap_index_0 = incoming_index
                    else:
                        heap_value_1 = incoming_value
                        heap_index_1 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    heap_value_1 = heap_value_0
                    heap_index_1 = heap_index_0
                    heap_value_0 = incoming_value
                    heap_index_0 = incoming_index
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    pl.write(selected_values, [0, 0], heap_value_0)
                    pl.write(selected_indices, [0, 0], heap_index_0)
                    pl.write(selected_values, [0, 1], heap_value_1)
                    pl.write(selected_indices, [0, 1], heap_index_1)
                    pl.write(selected_values, [0, 2], heap_value_2)
                    pl.write(selected_indices, [0, 2], heap_index_2)
                    pl.write(selected_values, [0, 3], heap_value_3)
                    pl.write(selected_indices, [0, 3], heap_index_3)
                    pl.write(selected_values, [0, 4], heap_value_4)
                    pl.write(selected_indices, [0, 4], heap_index_4)
                    pl.write(selected_values, [0, 5], heap_value_5)
                    pl.write(selected_indices, [0, 5], heap_index_5)
                    pl.write(selected_values, [0, 6], heap_value_6)
                    pl.write(selected_indices, [0, 6], heap_index_6)
                    pl.write(selected_values, [0, 7], heap_value_7)
                    pl.write(selected_indices, [0, 7], heap_index_7)
                    selected_sum = pl.read(selected_values, [0, 0])
                    for slot in pl.unroll(1, 8):
                        selected_sum = selected_sum + pl.read(selected_values, [0, slot])
                    weights = pl.div(selected_values, selected_sum + 1e-08)
                    for chunk in pl.range(cols // CHUNK_WIDTH):
                        column = chunk * CHUNK_WIDTH
                        input_values = pl.slice(logits, [1, CHUNK_WIDTH], [row, column])
                        values = pl.cast(input_values, pl.FP32)
                        probabilities = pl.div(pl.exp(pl.sub(values, maximum)), denominator)
                        for slot in pl.unroll(8):
                            winner = pl.read(selected_indices, [0, slot])
                            chunk_column32 = pl.cast(column, pl.INT32)
                            relative_winner = winner - chunk_column32
                            if relative_winner >= 0 and relative_winner < CHUNK_WIDTH:
                                local_winner = pl.cast(relative_winner, pl.INDEX)
                                original_probability = pl.read(probabilities, [0, local_winner])
                                pl.write(probabilities, [0, local_winner], original_probability - 2.0)
                        p = pl.assemble(p, probabilities, [row, column])
                    topk_w = pl.assemble(topk_w, weights, [row, 0])
                    topk_idx = pl.assemble(topk_idx, selected_indices, [row, 0])
    return (topk_w, topk_idx, p)

@pl.jit
def bounded_fp32(logits: pl.Tensor, colidx: pl.Tensor, p: pl.Out[pl.Tensor], topk_w: pl.Out[pl.Tensor], topk_idx: pl.Out[pl.Tensor]):
    rows, cols = logits.shape
    for row_base in pl.parallel(0, rows, ROWS_PER_TASK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='capacity_bounded_gating'):
            for row_offset in pl.range(ROWS_PER_TASK):
                row = row_base + row_offset
                if row < rows:
                    maximum_input = pl.slice(logits, [1, CHUNK_WIDTH], [row, 0])
                    maximum_values = maximum_input
                    maximum_lanes = pl.col_max(pl.reshape(maximum_values, [CHUNK_WIDTH // 8, 8]))
                    maximum = pl.read(maximum_lanes, [0, 0])
                    for lane in pl.unroll(1, 8):
                        lane_value = pl.read(maximum_lanes, [0, lane])
                        if lane_value > maximum:
                            maximum = lane_value
                    for chunk in pl.range(1, cols // CHUNK_WIDTH):
                        column = chunk * CHUNK_WIDTH
                        input_values = pl.slice(logits, [1, CHUNK_WIDTH], [row, column])
                        values = input_values
                        chunk_max_lanes = pl.col_max(pl.reshape(values, [CHUNK_WIDTH // 8, 8]))
                        chunk_max = pl.read(chunk_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(chunk_max_lanes, [0, lane])
                            if lane_value > chunk_max:
                                chunk_max = lane_value
                        if chunk_max > maximum:
                            maximum = chunk_max
                    denominator = 0.0
                    for chunk in pl.range(cols // CHUNK_WIDTH):
                        column = chunk * CHUNK_WIDTH
                        input_values = pl.slice(logits, [1, CHUNK_WIDTH], [row, column])
                        values = input_values
                        exponentials = pl.exp(pl.sub(values, maximum))
                        chunk_sum_lanes = pl.col_sum(pl.reshape(exponentials, [CHUNK_WIDTH // 8, 8]))
                        chunk_sum = pl.read(chunk_sum_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(chunk_sum_lanes, [0, lane])
                            chunk_sum = chunk_sum + lane_value
                        denominator = denominator + chunk_sum
                    first_input = pl.slice(logits, [1, CHUNK_WIDTH], [row, 0])
                    first_values = first_input
                    probabilities = pl.div(pl.exp(pl.sub(first_values, maximum)), denominator)
                    heap_value_0 = pl.read(probabilities, [0, 0])
                    heap_index_0 = pl.cast(0, pl.INT32)
                    heap_value_1 = pl.read(probabilities, [0, 1])
                    heap_index_1 = pl.cast(1, pl.INT32)
                    heap_value_2 = pl.read(probabilities, [0, 2])
                    heap_index_2 = pl.cast(2, pl.INT32)
                    heap_value_3 = pl.read(probabilities, [0, 3])
                    heap_index_3 = pl.cast(3, pl.INT32)
                    heap_value_4 = pl.read(probabilities, [0, 4])
                    heap_index_4 = pl.cast(4, pl.INT32)
                    heap_value_5 = pl.read(probabilities, [0, 5])
                    heap_index_5 = pl.cast(5, pl.INT32)
                    heap_value_6 = pl.read(probabilities, [0, 6])
                    heap_index_6 = pl.cast(6, pl.INT32)
                    heap_value_7 = pl.read(probabilities, [0, 7])
                    heap_index_7 = pl.cast(7, pl.INT32)
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_7
                    heap_index_3 = heap_index_7
                    if heap_value_3 > incoming_value:
                        heap_value_7 = heap_value_3
                        heap_index_7 = heap_index_3
                        heap_value_3 = incoming_value
                        heap_index_3 = incoming_index
                    else:
                        heap_value_7 = incoming_value
                        heap_index_7 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    if heap_value_6 > heap_value_5:
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    else:
                        heap_value_2 = heap_value_6
                        heap_index_2 = heap_index_6
                        if heap_value_2 > incoming_value:
                            heap_value_6 = heap_value_2
                            heap_index_6 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_6 = incoming_value
                            heap_index_6 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    if heap_value_4 > heap_value_3:
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        heap_value_3 = heap_value_7
                        heap_index_3 = heap_index_7
                        if heap_value_3 > incoming_value:
                            heap_value_7 = heap_value_3
                            heap_index_7 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_7 = incoming_value
                            heap_index_7 = incoming_index
                    else:
                        heap_value_1 = heap_value_4
                        heap_index_1 = heap_index_4
                        if heap_value_1 > incoming_value:
                            heap_value_4 = heap_value_1
                            heap_index_4 = heap_index_1
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                        else:
                            heap_value_4 = incoming_value
                            heap_index_4 = incoming_index
                    incoming_value = heap_value_0
                    incoming_index = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            heap_value_3 = heap_value_7
                            heap_index_3 = heap_index_7
                            if heap_value_3 > incoming_value:
                                heap_value_7 = heap_value_3
                                heap_index_7 = heap_index_3
                                if heap_value_1 > incoming_value:
                                    heap_value_3 = heap_value_1
                                    heap_index_3 = heap_index_1
                                    if heap_value_0 > incoming_value:
                                        heap_value_1 = heap_value_0
                                        heap_index_1 = heap_index_0
                                        heap_value_0 = incoming_value
                                        heap_index_0 = incoming_index
                                    else:
                                        heap_value_1 = incoming_value
                                        heap_index_1 = incoming_index
                                else:
                                    heap_value_3 = incoming_value
                                    heap_index_3 = incoming_index
                            else:
                                heap_value_7 = incoming_value
                                heap_index_7 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    for block in pl.range(cols // BLOCK_WIDTH):
                        column = block * BLOCK_WIDTH
                        selection_input = pl.slice(logits, [1, BLOCK_WIDTH], [row, column])
                        selection_values = selection_input
                        block_probabilities = pl.div(pl.exp(pl.sub(selection_values, maximum)), denominator)
                        block_max_lanes = pl.col_max(pl.reshape(block_probabilities, [BLOCK_WIDTH // 8, 8]))
                        block_max = pl.read(block_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(block_max_lanes, [0, lane])
                            if lane_value > block_max:
                                block_max = lane_value
                        if block_max > heap_value_0:
                            for block_offset in pl.range(BLOCK_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        if heap_value_2 > heap_value_1:
                                            heap_value_0 = heap_value_1
                                            heap_index_0 = heap_index_1
                                            if heap_value_4 > heap_value_3:
                                                heap_value_1 = heap_value_3
                                                heap_index_1 = heap_index_3
                                                heap_value_3 = heap_value_7
                                                heap_index_3 = heap_index_7
                                                if heap_value_3 > incoming_value:
                                                    heap_value_7 = heap_value_3
                                                    heap_index_7 = heap_index_3
                                                    if heap_value_1 > incoming_value:
                                                        heap_value_3 = heap_value_1
                                                        heap_index_3 = heap_index_1
                                                        if heap_value_0 > incoming_value:
                                                            heap_value_1 = heap_value_0
                                                            heap_index_1 = heap_index_0
                                                            heap_value_0 = incoming_value
                                                            heap_index_0 = incoming_index
                                                        else:
                                                            heap_value_1 = incoming_value
                                                            heap_index_1 = incoming_index
                                                    else:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                else:
                                                    heap_value_7 = incoming_value
                                                    heap_index_7 = incoming_index
                                            else:
                                                heap_value_1 = heap_value_4
                                                heap_index_1 = heap_index_4
                                                if heap_value_1 > incoming_value:
                                                    heap_value_4 = heap_value_1
                                                    heap_index_4 = heap_index_1
                                                    if heap_value_0 > incoming_value:
                                                        heap_value_1 = heap_value_0
                                                        heap_index_1 = heap_index_0
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    else:
                                                        heap_value_1 = incoming_value
                                                        heap_index_1 = incoming_index
                                                else:
                                                    heap_value_4 = incoming_value
                                                    heap_index_4 = incoming_index
                                        else:
                                            heap_value_0 = heap_value_2
                                            heap_index_0 = heap_index_2
                                            if heap_value_6 > heap_value_5:
                                                heap_value_2 = heap_value_5
                                                heap_index_2 = heap_index_5
                                                if heap_value_2 > incoming_value:
                                                    heap_value_5 = heap_value_2
                                                    heap_index_5 = heap_index_2
                                                    if heap_value_0 > incoming_value:
                                                        heap_value_2 = heap_value_0
                                                        heap_index_2 = heap_index_0
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    else:
                                                        heap_value_2 = incoming_value
                                                        heap_index_2 = incoming_index
                                                else:
                                                    heap_value_5 = incoming_value
                                                    heap_index_5 = incoming_index
                                            else:
                                                heap_value_2 = heap_value_6
                                                heap_index_2 = heap_index_6
                                                if heap_value_2 > incoming_value:
                                                    heap_value_6 = heap_value_2
                                                    heap_index_6 = heap_index_2
                                                    if heap_value_0 > incoming_value:
                                                        heap_value_2 = heap_value_0
                                                        heap_index_2 = heap_index_0
                                                        heap_value_0 = incoming_value
                                                        heap_index_0 = incoming_index
                                                    else:
                                                        heap_value_2 = incoming_value
                                                        heap_index_2 = incoming_index
                                                else:
                                                    heap_value_6 = incoming_value
                                                    heap_index_6 = incoming_index
                    incoming_value = heap_value_7
                    incoming_index = heap_index_7
                    heap_value_7 = heap_value_0
                    heap_index_7 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    incoming_value = heap_value_6
                    incoming_index = heap_index_6
                    heap_value_6 = heap_value_0
                    heap_index_6 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            if heap_value_0 > incoming_value:
                                heap_value_2 = heap_value_0
                                heap_index_2 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_2 = incoming_value
                                heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    incoming_value = heap_value_5
                    incoming_index = heap_index_5
                    heap_value_5 = heap_value_0
                    heap_index_5 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_4
                    incoming_index = heap_index_4
                    heap_value_4 = heap_value_0
                    heap_index_4 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        if heap_value_1 > incoming_value:
                            heap_value_3 = heap_value_1
                            heap_index_3 = heap_index_1
                            if heap_value_0 > incoming_value:
                                heap_value_1 = heap_value_0
                                heap_index_1 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                        else:
                            heap_value_3 = incoming_value
                            heap_index_3 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_0
                    heap_index_3 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_0 > incoming_value:
                            heap_value_1 = heap_value_0
                            heap_index_1 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    heap_value_2 = heap_value_0
                    heap_index_2 = heap_index_0
                    heap_value_0 = heap_value_1
                    heap_index_0 = heap_index_1
                    if heap_value_0 > incoming_value:
                        heap_value_1 = heap_value_0
                        heap_index_1 = heap_index_0
                        heap_value_0 = incoming_value
                        heap_index_0 = incoming_index
                    else:
                        heap_value_1 = incoming_value
                        heap_index_1 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    heap_value_1 = heap_value_0
                    heap_index_1 = heap_index_0
                    heap_value_0 = incoming_value
                    heap_index_0 = incoming_index
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    pl.write(selected_values, [0, 0], heap_value_0)
                    pl.write(selected_indices, [0, 0], heap_index_0)
                    pl.write(selected_values, [0, 1], heap_value_1)
                    pl.write(selected_indices, [0, 1], heap_index_1)
                    pl.write(selected_values, [0, 2], heap_value_2)
                    pl.write(selected_indices, [0, 2], heap_index_2)
                    pl.write(selected_values, [0, 3], heap_value_3)
                    pl.write(selected_indices, [0, 3], heap_index_3)
                    pl.write(selected_values, [0, 4], heap_value_4)
                    pl.write(selected_indices, [0, 4], heap_index_4)
                    pl.write(selected_values, [0, 5], heap_value_5)
                    pl.write(selected_indices, [0, 5], heap_index_5)
                    pl.write(selected_values, [0, 6], heap_value_6)
                    pl.write(selected_indices, [0, 6], heap_index_6)
                    pl.write(selected_values, [0, 7], heap_value_7)
                    pl.write(selected_indices, [0, 7], heap_index_7)
                    selected_sum = pl.read(selected_values, [0, 0])
                    for slot in pl.unroll(1, 8):
                        selected_sum = selected_sum + pl.read(selected_values, [0, slot])
                    weights = pl.div(selected_values, selected_sum + 1e-08)
                    for chunk in pl.range(cols // CHUNK_WIDTH):
                        column = chunk * CHUNK_WIDTH
                        input_values = pl.slice(logits, [1, CHUNK_WIDTH], [row, column])
                        values = input_values
                        probabilities = pl.div(pl.exp(pl.sub(values, maximum)), denominator)
                        for slot in pl.unroll(8):
                            winner = pl.read(selected_indices, [0, slot])
                            chunk_column32 = pl.cast(column, pl.INT32)
                            relative_winner = winner - chunk_column32
                            if relative_winner >= 0 and relative_winner < CHUNK_WIDTH:
                                local_winner = pl.cast(relative_winner, pl.INDEX)
                                original_probability = pl.read(probabilities, [0, local_winner])
                                pl.write(probabilities, [0, local_winner], original_probability - 2.0)
                        p = pl.assemble(p, probabilities, [row, column])
                    topk_w = pl.assemble(topk_w, weights, [row, 0])
                    topk_idx = pl.assemble(topk_idx, selected_indices, [row, 0])
    return (topk_w, topk_idx, p)

@pl.jit
def flat_resident(logits: pl.Tensor, colidx: pl.Tensor, p: pl.Out[pl.Tensor], topk_w: pl.Out[pl.Tensor], topk_idx: pl.Out[pl.Tensor]):
    rows, cols = logits.shape
    for row_block in pl.spmd((rows + ROWS_PER_TASK - 1) // ROWS_PER_TASK, name_hint='capacity_resident_gating'):
        row_base = row_block * ROWS_PER_TASK
        for row_offset in pl.range(ROWS_PER_TASK):
            row = row_base + row_offset
            if row < rows:
                input_values = pl.slice(logits, [1, cols], [row, 0])
                values = pl.cast(input_values, pl.FP32)
                maximum_lanes = pl.col_max(pl.reshape(values, [cols // 8, 8]))
                maximum = pl.read(maximum_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(maximum_lanes, [0, lane])
                    if lane_value > maximum:
                        maximum = lane_value
                exponentials = pl.exp(pl.sub(values, maximum))
                denominator_lanes = pl.col_sum(pl.reshape(exponentials, [cols // 8, 8]))
                denominator = pl.read(denominator_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(denominator_lanes, [0, lane])
                    denominator = denominator + lane_value
                probabilities = pl.div(exponentials, denominator)
                if cols < 512:
                    columns = pl.slice(colidx, [1, cols], [0, 0])
                    partition_values = pl.add(probabilities, 0.0)
                    partition_indices = pl.cast(columns, pl.INT32)
                    first = 0
                    last = cols
                    depth_limit = NARROW_DEPTH
                    for bounded_step_8 in pl.range(NARROW_DEPTH + 1):
                        if last - first > 3:
                            if depth_limit == 0:
                                heap_length = 8 - first
                                selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                                selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                                for slot in pl.range(heap_length):
                                    initial_value = pl.read(partition_values, [0, first + slot])
                                    initial_index = pl.read(partition_indices, [0, first + slot])
                                    pl.write(selected_values, [0, slot], initial_value)
                                    pl.write(selected_indices, [0, slot], initial_index)
                                if heap_length > 1:
                                    for parent_step in pl.range((heap_length - 2) // 2 + 1):
                                        hole = (heap_length - 2) // 2 - parent_step
                                        incoming_value = pl.read(selected_values, [0, hole])
                                        incoming_index = pl.read(selected_indices, [0, hole])
                                        hole_adjust_1 = hole
                                        top_adjust_1 = hole_adjust_1
                                        child_adjust_1 = hole_adjust_1
                                        for bounded_step_1 in pl.range(3):
                                            if child_adjust_1 < (heap_length - 1) // 2:
                                                child_adjust_1 = 2 * (child_adjust_1 + 1)
                                                right_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                left_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                                if right_value_adjust_1 > left_value_adjust_1:
                                                    child_adjust_1 = child_adjust_1 - 1
                                                moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1])
                                                pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                hole_adjust_1 = child_adjust_1
                                        if heap_length % 2 == 0 and child_adjust_1 == (heap_length - 2) // 2:
                                            child_adjust_1 = 2 * (child_adjust_1 + 1)
                                            moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                            moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1 - 1])
                                            pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                            pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                            hole_adjust_1 = child_adjust_1 - 1
                                        for bounded_step_2 in pl.range(3):
                                            if hole_adjust_1 > top_adjust_1:
                                                parent_adjust_1 = (hole_adjust_1 - 1) // 2
                                                parent_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                if not parent_value_adjust_1 <= incoming_value:
                                                    moving_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                    moving_index_adjust_1 = pl.read(selected_indices, [0, parent_adjust_1])
                                                    pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                    pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                    hole_adjust_1 = parent_adjust_1
                                        pl.write(selected_values, [0, hole_adjust_1], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_1], incoming_index)
                                for scan_column in pl.range(8, last):
                                    incoming_value = pl.read(partition_values, [0, scan_column])
                                    root = pl.read(selected_values, [0, 0])
                                    if incoming_value > root:
                                        incoming_index = pl.read(partition_indices, [0, scan_column])
                                        hole = 0
                                        hole_adjust_2 = hole
                                        top_adjust_2 = hole_adjust_2
                                        child_adjust_2 = hole_adjust_2
                                        for bounded_step_3 in pl.range(3):
                                            if child_adjust_2 < (heap_length - 1) // 2:
                                                child_adjust_2 = 2 * (child_adjust_2 + 1)
                                                right_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                left_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                                if right_value_adjust_2 > left_value_adjust_2:
                                                    child_adjust_2 = child_adjust_2 - 1
                                                moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2])
                                                pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                hole_adjust_2 = child_adjust_2
                                        if heap_length % 2 == 0 and child_adjust_2 == (heap_length - 2) // 2:
                                            child_adjust_2 = 2 * (child_adjust_2 + 1)
                                            moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                            moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2 - 1])
                                            pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                            pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                            hole_adjust_2 = child_adjust_2 - 1
                                        for bounded_step_4 in pl.range(3):
                                            if hole_adjust_2 > top_adjust_2:
                                                parent_adjust_2 = (hole_adjust_2 - 1) // 2
                                                parent_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                if not parent_value_adjust_2 <= incoming_value:
                                                    moving_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                    moving_index_adjust_2 = pl.read(selected_indices, [0, parent_adjust_2])
                                                    pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                    pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                    hole_adjust_2 = parent_adjust_2
                                        pl.write(selected_values, [0, hole_adjust_2], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_2], incoming_index)
                                for slot in pl.range(heap_length):
                                    copied_value = pl.read(selected_values, [0, slot])
                                    copied_index = pl.read(selected_indices, [0, slot])
                                    pl.write(partition_values, [0, first + slot], copied_value)
                                    pl.write(partition_indices, [0, first + slot], copied_index)
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, 7])
                                moving_index = pl.read(partition_indices, [0, 7])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, 7], swap_value)
                                pl.write(partition_indices, [0, 7], swap_index)
                                first = last
                            else:
                                depth_limit = depth_limit - 1
                                median_left = first + 1
                                median_middle = first + (last - first) // 2
                                median_right = last - 1
                                left_value = pl.read(partition_values, [0, median_left])
                                middle_value = pl.read(partition_values, [0, median_middle])
                                right_value = pl.read(partition_values, [0, median_right])
                                pivot_column = median_middle
                                if left_value > middle_value:
                                    if middle_value > right_value:
                                        pivot_column = median_middle
                                    elif left_value > right_value:
                                        pivot_column = median_right
                                    else:
                                        pivot_column = median_left
                                elif left_value > right_value:
                                    pivot_column = median_left
                                elif middle_value > right_value:
                                    pivot_column = median_right
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, pivot_column])
                                moving_index = pl.read(partition_indices, [0, pivot_column])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, pivot_column], swap_value)
                                pl.write(partition_indices, [0, pivot_column], swap_index)
                                pivot_value = pl.read(partition_values, [0, first])
                                partition_left = first + 1
                                partition_right = last
                                partition_active = True
                                for bounded_step_7 in pl.range(cols + 8):
                                    if partition_active:
                                        probe_left = pl.read(partition_values, [0, partition_left])
                                        for bounded_step_5 in pl.range(cols + 8):
                                            if probe_left > pivot_value:
                                                partition_left = partition_left + 1
                                                probe_left = pl.read(partition_values, [0, partition_left])
                                        partition_right = partition_right - 1
                                        probe_right = pl.read(partition_values, [0, partition_right])
                                        for bounded_step_6 in pl.range(cols + 8):
                                            if pivot_value > probe_right:
                                                partition_right = partition_right - 1
                                                probe_right = pl.read(partition_values, [0, partition_right])
                                        if partition_left >= partition_right:
                                            partition_active = False
                                        else:
                                            swap_value = pl.read(partition_values, [0, partition_left])
                                            swap_index = pl.read(partition_indices, [0, partition_left])
                                            moving_value = pl.read(partition_values, [0, partition_right])
                                            moving_index = pl.read(partition_indices, [0, partition_right])
                                            pl.write(partition_values, [0, partition_left], moving_value)
                                            pl.write(partition_indices, [0, partition_left], moving_index)
                                            pl.write(partition_values, [0, partition_right], swap_value)
                                            pl.write(partition_indices, [0, partition_right], swap_index)
                                            partition_left = partition_left + 1
                                if partition_left <= 7:
                                    first = partition_left
                                else:
                                    last = partition_left
                    for insertion_column in pl.range(first + 1, last):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_9 in pl.range(cols + 8):
                            if insertion_hole > first:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    for insertion_column in pl.range(0 + 1, 7):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_10 in pl.range(cols + 8):
                            if insertion_hole > 0:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    for slot in pl.range(8):
                        copied_value = pl.read(partition_values, [0, slot])
                        copied_index = pl.read(partition_indices, [0, slot])
                        pl.write(selected_values, [0, slot], copied_value)
                        pl.write(selected_indices, [0, slot], copied_index)
                else:
                    heap_value_0 = pl.read(probabilities, [0, 0])
                    heap_index_0 = pl.cast(0, pl.INT32)
                    heap_value_1 = pl.read(probabilities, [0, 1])
                    heap_index_1 = pl.cast(1, pl.INT32)
                    heap_value_2 = pl.read(probabilities, [0, 2])
                    heap_index_2 = pl.cast(2, pl.INT32)
                    heap_value_3 = pl.read(probabilities, [0, 3])
                    heap_index_3 = pl.cast(3, pl.INT32)
                    heap_value_4 = pl.read(probabilities, [0, 4])
                    heap_index_4 = pl.cast(4, pl.INT32)
                    heap_value_5 = pl.read(probabilities, [0, 5])
                    heap_index_5 = pl.cast(5, pl.INT32)
                    heap_value_6 = pl.read(probabilities, [0, 6])
                    heap_index_6 = pl.cast(6, pl.INT32)
                    heap_value_7 = pl.read(probabilities, [0, 7])
                    heap_index_7 = pl.cast(7, pl.INT32)
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_7
                    heap_index_3 = heap_index_7
                    if heap_value_3 > incoming_value:
                        heap_value_7 = heap_value_3
                        heap_index_7 = heap_index_3
                        heap_value_3 = incoming_value
                        heap_index_3 = incoming_index
                    else:
                        heap_value_7 = incoming_value
                        heap_index_7 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    if heap_value_6 > heap_value_5:
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    else:
                        heap_value_2 = heap_value_6
                        heap_index_2 = heap_index_6
                        if heap_value_2 > incoming_value:
                            heap_value_6 = heap_value_2
                            heap_index_6 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_6 = incoming_value
                            heap_index_6 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    if heap_value_4 > heap_value_3:
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        heap_value_3 = heap_value_7
                        heap_index_3 = heap_index_7
                        if heap_value_3 > incoming_value:
                            heap_value_7 = heap_value_3
                            heap_index_7 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_7 = incoming_value
                            heap_index_7 = incoming_index
                    else:
                        heap_value_1 = heap_value_4
                        heap_index_1 = heap_index_4
                        if heap_value_1 > incoming_value:
                            heap_value_4 = heap_value_1
                            heap_index_4 = heap_index_1
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                        else:
                            heap_value_4 = incoming_value
                            heap_index_4 = incoming_index
                    incoming_value = heap_value_0
                    incoming_index = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            heap_value_3 = heap_value_7
                            heap_index_3 = heap_index_7
                            if heap_value_3 > incoming_value:
                                heap_value_7 = heap_value_3
                                heap_index_7 = heap_index_3
                                if heap_value_1 > incoming_value:
                                    heap_value_3 = heap_value_1
                                    heap_index_3 = heap_index_1
                                    if heap_value_0 > incoming_value:
                                        heap_value_1 = heap_value_0
                                        heap_index_1 = heap_index_0
                                        heap_value_0 = incoming_value
                                        heap_index_0 = incoming_index
                                    else:
                                        heap_value_1 = incoming_value
                                        heap_index_1 = incoming_index
                                else:
                                    heap_value_3 = incoming_value
                                    heap_index_3 = incoming_index
                            else:
                                heap_value_7 = incoming_value
                                heap_index_7 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    summary_prefix = pl.slice(probabilities, [1, RESIDENT_SUMMARY_PREFIX_WIDTH], [0, 0])
                    summary_values = pl.reshape(summary_prefix, [RESIDENT_SUMMARY_PREFIX_WIDTH // RESIDENT_BLOCK_WIDTH, RESIDENT_BLOCK_WIDTH])
                    block_maxima = pl.row_max(summary_values)
                    for block in pl.range(cols // RESIDENT_BLOCK_WIDTH):
                        column = block * RESIDENT_BLOCK_WIDTH
                        block_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                        if block < RESIDENT_SUMMARY_BLOCKS:
                            block_max = pl.read(block_maxima, [block, 0])
                        else:
                            block_max_lanes = pl.col_max(pl.reshape(block_probabilities, [RESIDENT_BLOCK_WIDTH // 8, 8]))
                            block_max = pl.read(block_max_lanes, [0, 0])
                            for lane in pl.unroll(1, 8):
                                lane_value = pl.read(block_max_lanes, [0, lane])
                                if lane_value > block_max:
                                    block_max = lane_value
                        if block_max > heap_value_0:
                            for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    root_flat_resident_0_path_first = 0
                                    root_flat_resident_0_path_first_value = 0.0
                                    root_flat_resident_0_path_first_index = pl.cast(0, pl.INT32)
                                    root_flat_resident_0_path_second = 0
                                    root_flat_resident_0_path_second_value = 0.0
                                    root_flat_resident_0_path_second_index = pl.cast(0, pl.INT32)
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        root_flat_resident_0_path_first = 2
                                        root_flat_resident_0_path_first_value = heap_value_2
                                        root_flat_resident_0_path_first_index = heap_index_2
                                        if heap_value_2 > heap_value_1:
                                            root_flat_resident_0_path_first = 1
                                            root_flat_resident_0_path_first_value = heap_value_1
                                            root_flat_resident_0_path_first_index = heap_index_1
                                        if root_flat_resident_0_path_first_value > incoming_value:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        else:
                                            heap_value_0 = root_flat_resident_0_path_first_value
                                            heap_index_0 = root_flat_resident_0_path_first_index
                                            root_flat_resident_0_path_second = 6
                                            root_flat_resident_0_path_second_value = heap_value_6
                                            root_flat_resident_0_path_second_index = heap_index_6
                                            if root_flat_resident_0_path_first == 1:
                                                root_flat_resident_0_path_second = 4
                                                root_flat_resident_0_path_second_value = heap_value_4
                                                root_flat_resident_0_path_second_index = heap_index_4
                                                if heap_value_4 > heap_value_3:
                                                    root_flat_resident_0_path_second = 3
                                                    root_flat_resident_0_path_second_value = heap_value_3
                                                    root_flat_resident_0_path_second_index = heap_index_3
                                            elif heap_value_6 > heap_value_5:
                                                root_flat_resident_0_path_second = 5
                                                root_flat_resident_0_path_second_value = heap_value_5
                                                root_flat_resident_0_path_second_index = heap_index_5
                                            if root_flat_resident_0_path_second_value > incoming_value:
                                                if root_flat_resident_0_path_first == 1:
                                                    heap_value_1 = incoming_value
                                                    heap_index_1 = incoming_index
                                                else:
                                                    heap_value_2 = incoming_value
                                                    heap_index_2 = incoming_index
                                            else:
                                                if root_flat_resident_0_path_first == 1:
                                                    heap_value_1 = root_flat_resident_0_path_second_value
                                                    heap_index_1 = root_flat_resident_0_path_second_index
                                                else:
                                                    heap_value_2 = root_flat_resident_0_path_second_value
                                                    heap_index_2 = root_flat_resident_0_path_second_index
                                                if root_flat_resident_0_path_second == 3:
                                                    if heap_value_7 > incoming_value:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                    else:
                                                        heap_value_3 = heap_value_7
                                                        heap_index_3 = heap_index_7
                                                        heap_value_7 = incoming_value
                                                        heap_index_7 = incoming_index
                                                elif root_flat_resident_0_path_second == 4:
                                                    heap_value_4 = incoming_value
                                                    heap_index_4 = incoming_index
                                                elif root_flat_resident_0_path_second == 5:
                                                    heap_value_5 = incoming_value
                                                    heap_index_5 = incoming_index
                                                else:
                                                    heap_value_6 = incoming_value
                                                    heap_index_6 = incoming_index
                    if cols % RESIDENT_BLOCK_WIDTH != 0:
                        column = cols // RESIDENT_BLOCK_WIDTH * RESIDENT_BLOCK_WIDTH
                        block_probabilities = pl.slice(probabilities, [1, RESIDENT_TAIL_WIDTH], [0, column])
                        tail_max_lanes = pl.col_max(pl.reshape(block_probabilities, [RESIDENT_TAIL_WIDTH // 8, 8]))
                        tail_max = pl.read(tail_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(tail_max_lanes, [0, lane])
                            if lane_value > tail_max:
                                tail_max = lane_value
                        if tail_max > heap_value_0:
                            for block_offset in pl.range(RESIDENT_TAIL_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    root_flat_resident_1_path_first = 0
                                    root_flat_resident_1_path_first_value = 0.0
                                    root_flat_resident_1_path_first_index = pl.cast(0, pl.INT32)
                                    root_flat_resident_1_path_second = 0
                                    root_flat_resident_1_path_second_value = 0.0
                                    root_flat_resident_1_path_second_index = pl.cast(0, pl.INT32)
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        root_flat_resident_1_path_first = 2
                                        root_flat_resident_1_path_first_value = heap_value_2
                                        root_flat_resident_1_path_first_index = heap_index_2
                                        if heap_value_2 > heap_value_1:
                                            root_flat_resident_1_path_first = 1
                                            root_flat_resident_1_path_first_value = heap_value_1
                                            root_flat_resident_1_path_first_index = heap_index_1
                                        if root_flat_resident_1_path_first_value > incoming_value:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        else:
                                            heap_value_0 = root_flat_resident_1_path_first_value
                                            heap_index_0 = root_flat_resident_1_path_first_index
                                            root_flat_resident_1_path_second = 6
                                            root_flat_resident_1_path_second_value = heap_value_6
                                            root_flat_resident_1_path_second_index = heap_index_6
                                            if root_flat_resident_1_path_first == 1:
                                                root_flat_resident_1_path_second = 4
                                                root_flat_resident_1_path_second_value = heap_value_4
                                                root_flat_resident_1_path_second_index = heap_index_4
                                                if heap_value_4 > heap_value_3:
                                                    root_flat_resident_1_path_second = 3
                                                    root_flat_resident_1_path_second_value = heap_value_3
                                                    root_flat_resident_1_path_second_index = heap_index_3
                                            elif heap_value_6 > heap_value_5:
                                                root_flat_resident_1_path_second = 5
                                                root_flat_resident_1_path_second_value = heap_value_5
                                                root_flat_resident_1_path_second_index = heap_index_5
                                            if root_flat_resident_1_path_second_value > incoming_value:
                                                if root_flat_resident_1_path_first == 1:
                                                    heap_value_1 = incoming_value
                                                    heap_index_1 = incoming_index
                                                else:
                                                    heap_value_2 = incoming_value
                                                    heap_index_2 = incoming_index
                                            else:
                                                if root_flat_resident_1_path_first == 1:
                                                    heap_value_1 = root_flat_resident_1_path_second_value
                                                    heap_index_1 = root_flat_resident_1_path_second_index
                                                else:
                                                    heap_value_2 = root_flat_resident_1_path_second_value
                                                    heap_index_2 = root_flat_resident_1_path_second_index
                                                if root_flat_resident_1_path_second == 3:
                                                    if heap_value_7 > incoming_value:
                                                        heap_value_3 = incoming_value
                                                        heap_index_3 = incoming_index
                                                    else:
                                                        heap_value_3 = heap_value_7
                                                        heap_index_3 = heap_index_7
                                                        heap_value_7 = incoming_value
                                                        heap_index_7 = incoming_index
                                                elif root_flat_resident_1_path_second == 4:
                                                    heap_value_4 = incoming_value
                                                    heap_index_4 = incoming_index
                                                elif root_flat_resident_1_path_second == 5:
                                                    heap_value_5 = incoming_value
                                                    heap_index_5 = incoming_index
                                                else:
                                                    heap_value_6 = incoming_value
                                                    heap_index_6 = incoming_index
                    incoming_value = heap_value_7
                    incoming_index = heap_index_7
                    heap_value_7 = heap_value_0
                    heap_index_7 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    incoming_value = heap_value_6
                    incoming_index = heap_index_6
                    heap_value_6 = heap_value_0
                    heap_index_6 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            if heap_value_0 > incoming_value:
                                heap_value_2 = heap_value_0
                                heap_index_2 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_2 = incoming_value
                                heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    incoming_value = heap_value_5
                    incoming_index = heap_index_5
                    heap_value_5 = heap_value_0
                    heap_index_5 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_4
                    incoming_index = heap_index_4
                    heap_value_4 = heap_value_0
                    heap_index_4 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        if heap_value_1 > incoming_value:
                            heap_value_3 = heap_value_1
                            heap_index_3 = heap_index_1
                            if heap_value_0 > incoming_value:
                                heap_value_1 = heap_value_0
                                heap_index_1 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                        else:
                            heap_value_3 = incoming_value
                            heap_index_3 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_0
                    heap_index_3 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_0 > incoming_value:
                            heap_value_1 = heap_value_0
                            heap_index_1 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    heap_value_2 = heap_value_0
                    heap_index_2 = heap_index_0
                    heap_value_0 = heap_value_1
                    heap_index_0 = heap_index_1
                    if heap_value_0 > incoming_value:
                        heap_value_1 = heap_value_0
                        heap_index_1 = heap_index_0
                        heap_value_0 = incoming_value
                        heap_index_0 = incoming_index
                    else:
                        heap_value_1 = incoming_value
                        heap_index_1 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    heap_value_1 = heap_value_0
                    heap_index_1 = heap_index_0
                    heap_value_0 = incoming_value
                    heap_index_0 = incoming_index
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    pl.write(selected_values, [0, 0], heap_value_0)
                    pl.write(selected_indices, [0, 0], heap_index_0)
                    pl.write(selected_values, [0, 1], heap_value_1)
                    pl.write(selected_indices, [0, 1], heap_index_1)
                    pl.write(selected_values, [0, 2], heap_value_2)
                    pl.write(selected_indices, [0, 2], heap_index_2)
                    pl.write(selected_values, [0, 3], heap_value_3)
                    pl.write(selected_indices, [0, 3], heap_index_3)
                    pl.write(selected_values, [0, 4], heap_value_4)
                    pl.write(selected_indices, [0, 4], heap_index_4)
                    pl.write(selected_values, [0, 5], heap_value_5)
                    pl.write(selected_indices, [0, 5], heap_index_5)
                    pl.write(selected_values, [0, 6], heap_value_6)
                    pl.write(selected_indices, [0, 6], heap_index_6)
                    pl.write(selected_values, [0, 7], heap_value_7)
                    pl.write(selected_indices, [0, 7], heap_index_7)
                selected_sum = pl.read(selected_values, [0, 0])
                for slot in pl.unroll(1, 8):
                    selected_sum = selected_sum + pl.read(selected_values, [0, slot])
                weights = pl.div(selected_values, selected_sum + 1e-08)
                for slot in pl.unroll(8):
                    winner = pl.cast(pl.read(selected_indices, [0, slot]), pl.INDEX)
                    original_probability = pl.read(probabilities, [0, winner])
                    pl.write(probabilities, [0, winner], original_probability - 2.0)
                p = pl.assemble(p, probabilities, [row, 0])
                topk_w = pl.assemble(topk_w, weights, [row, 0])
                topk_idx = pl.assemble(topk_idx, selected_indices, [row, 0])
    return (topk_w, topk_idx, p)


@pl.jit
def flat_resident_fp32(logits: pl.Tensor, colidx: pl.Tensor, p: pl.Out[pl.Tensor], topk_w: pl.Out[pl.Tensor], topk_idx: pl.Out[pl.Tensor]):
    rows, cols = logits.shape
    for row_block in pl.spmd((rows + ROWS_PER_TASK - 1) // ROWS_PER_TASK, name_hint='capacity_resident_gating'):
        row_base = row_block * ROWS_PER_TASK
        for row_offset in pl.range(ROWS_PER_TASK):
            row = row_base + row_offset
            if row < rows:
                input_values = pl.slice(logits, [1, cols], [row, 0])
                values = input_values
                maximum_lanes = pl.col_max(pl.reshape(values, [cols // 8, 8]))
                maximum = pl.read(maximum_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(maximum_lanes, [0, lane])
                    if lane_value > maximum:
                        maximum = lane_value
                exponentials = pl.exp(pl.sub(values, maximum))
                denominator_lanes = pl.col_sum(pl.reshape(exponentials, [cols // 8, 8]))
                denominator = pl.read(denominator_lanes, [0, 0])
                for lane in pl.unroll(1, 8):
                    lane_value = pl.read(denominator_lanes, [0, lane])
                    denominator = denominator + lane_value
                probabilities = pl.div(exponentials, denominator)
                if cols < 512:
                    columns = pl.slice(colidx, [1, cols], [0, 0])
                    partition_values = pl.add(probabilities, 0.0)
                    partition_indices = pl.cast(columns, pl.INT32)
                    first = 0
                    last = cols
                    depth_limit = NARROW_DEPTH
                    for bounded_step_18 in pl.range(NARROW_DEPTH + 1):
                        if last - first > 3:
                            if depth_limit == 0:
                                heap_length = 8 - first
                                selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                                selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                                for slot in pl.range(heap_length):
                                    initial_value = pl.read(partition_values, [0, first + slot])
                                    initial_index = pl.read(partition_indices, [0, first + slot])
                                    pl.write(selected_values, [0, slot], initial_value)
                                    pl.write(selected_indices, [0, slot], initial_index)
                                if heap_length > 1:
                                    for parent_step in pl.range((heap_length - 2) // 2 + 1):
                                        hole = (heap_length - 2) // 2 - parent_step
                                        incoming_value = pl.read(selected_values, [0, hole])
                                        incoming_index = pl.read(selected_indices, [0, hole])
                                        hole_adjust_1 = hole
                                        top_adjust_1 = hole_adjust_1
                                        child_adjust_1 = hole_adjust_1
                                        for bounded_step_11 in pl.range(3):
                                            if child_adjust_1 < (heap_length - 1) // 2:
                                                child_adjust_1 = 2 * (child_adjust_1 + 1)
                                                right_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                left_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                                if right_value_adjust_1 > left_value_adjust_1:
                                                    child_adjust_1 = child_adjust_1 - 1
                                                moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1])
                                                moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1])
                                                pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                hole_adjust_1 = child_adjust_1
                                        if heap_length % 2 == 0 and child_adjust_1 == (heap_length - 2) // 2:
                                            child_adjust_1 = 2 * (child_adjust_1 + 1)
                                            moving_value_adjust_1 = pl.read(selected_values, [0, child_adjust_1 - 1])
                                            moving_index_adjust_1 = pl.read(selected_indices, [0, child_adjust_1 - 1])
                                            pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                            pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                            hole_adjust_1 = child_adjust_1 - 1
                                        for bounded_step_12 in pl.range(3):
                                            if hole_adjust_1 > top_adjust_1:
                                                parent_adjust_1 = (hole_adjust_1 - 1) // 2
                                                parent_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                if not parent_value_adjust_1 <= incoming_value:
                                                    moving_value_adjust_1 = pl.read(selected_values, [0, parent_adjust_1])
                                                    moving_index_adjust_1 = pl.read(selected_indices, [0, parent_adjust_1])
                                                    pl.write(selected_values, [0, hole_adjust_1], moving_value_adjust_1)
                                                    pl.write(selected_indices, [0, hole_adjust_1], moving_index_adjust_1)
                                                    hole_adjust_1 = parent_adjust_1
                                        pl.write(selected_values, [0, hole_adjust_1], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_1], incoming_index)
                                for scan_column in pl.range(8, last):
                                    incoming_value = pl.read(partition_values, [0, scan_column])
                                    root = pl.read(selected_values, [0, 0])
                                    if incoming_value > root:
                                        incoming_index = pl.read(partition_indices, [0, scan_column])
                                        hole = 0
                                        hole_adjust_2 = hole
                                        top_adjust_2 = hole_adjust_2
                                        child_adjust_2 = hole_adjust_2
                                        for bounded_step_13 in pl.range(3):
                                            if child_adjust_2 < (heap_length - 1) // 2:
                                                child_adjust_2 = 2 * (child_adjust_2 + 1)
                                                right_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                left_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                                if right_value_adjust_2 > left_value_adjust_2:
                                                    child_adjust_2 = child_adjust_2 - 1
                                                moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2])
                                                moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2])
                                                pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                hole_adjust_2 = child_adjust_2
                                        if heap_length % 2 == 0 and child_adjust_2 == (heap_length - 2) // 2:
                                            child_adjust_2 = 2 * (child_adjust_2 + 1)
                                            moving_value_adjust_2 = pl.read(selected_values, [0, child_adjust_2 - 1])
                                            moving_index_adjust_2 = pl.read(selected_indices, [0, child_adjust_2 - 1])
                                            pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                            pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                            hole_adjust_2 = child_adjust_2 - 1
                                        for bounded_step_14 in pl.range(3):
                                            if hole_adjust_2 > top_adjust_2:
                                                parent_adjust_2 = (hole_adjust_2 - 1) // 2
                                                parent_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                if not parent_value_adjust_2 <= incoming_value:
                                                    moving_value_adjust_2 = pl.read(selected_values, [0, parent_adjust_2])
                                                    moving_index_adjust_2 = pl.read(selected_indices, [0, parent_adjust_2])
                                                    pl.write(selected_values, [0, hole_adjust_2], moving_value_adjust_2)
                                                    pl.write(selected_indices, [0, hole_adjust_2], moving_index_adjust_2)
                                                    hole_adjust_2 = parent_adjust_2
                                        pl.write(selected_values, [0, hole_adjust_2], incoming_value)
                                        pl.write(selected_indices, [0, hole_adjust_2], incoming_index)
                                for slot in pl.range(heap_length):
                                    copied_value = pl.read(selected_values, [0, slot])
                                    copied_index = pl.read(selected_indices, [0, slot])
                                    pl.write(partition_values, [0, first + slot], copied_value)
                                    pl.write(partition_indices, [0, first + slot], copied_index)
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, 7])
                                moving_index = pl.read(partition_indices, [0, 7])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, 7], swap_value)
                                pl.write(partition_indices, [0, 7], swap_index)
                                first = last
                            else:
                                depth_limit = depth_limit - 1
                                median_left = first + 1
                                median_middle = first + (last - first) // 2
                                median_right = last - 1
                                left_value = pl.read(partition_values, [0, median_left])
                                middle_value = pl.read(partition_values, [0, median_middle])
                                right_value = pl.read(partition_values, [0, median_right])
                                pivot_column = median_middle
                                if left_value > middle_value:
                                    if middle_value > right_value:
                                        pivot_column = median_middle
                                    elif left_value > right_value:
                                        pivot_column = median_right
                                    else:
                                        pivot_column = median_left
                                elif left_value > right_value:
                                    pivot_column = median_left
                                elif middle_value > right_value:
                                    pivot_column = median_right
                                swap_value = pl.read(partition_values, [0, first])
                                swap_index = pl.read(partition_indices, [0, first])
                                moving_value = pl.read(partition_values, [0, pivot_column])
                                moving_index = pl.read(partition_indices, [0, pivot_column])
                                pl.write(partition_values, [0, first], moving_value)
                                pl.write(partition_indices, [0, first], moving_index)
                                pl.write(partition_values, [0, pivot_column], swap_value)
                                pl.write(partition_indices, [0, pivot_column], swap_index)
                                pivot_value = pl.read(partition_values, [0, first])
                                partition_left = first + 1
                                partition_right = last
                                partition_active = True
                                for bounded_step_17 in pl.range(cols + 8):
                                    if partition_active:
                                        probe_left = pl.read(partition_values, [0, partition_left])
                                        for bounded_step_15 in pl.range(cols + 8):
                                            if probe_left > pivot_value:
                                                partition_left = partition_left + 1
                                                probe_left = pl.read(partition_values, [0, partition_left])
                                        partition_right = partition_right - 1
                                        probe_right = pl.read(partition_values, [0, partition_right])
                                        for bounded_step_16 in pl.range(cols + 8):
                                            if pivot_value > probe_right:
                                                partition_right = partition_right - 1
                                                probe_right = pl.read(partition_values, [0, partition_right])
                                        if partition_left >= partition_right:
                                            partition_active = False
                                        else:
                                            swap_value = pl.read(partition_values, [0, partition_left])
                                            swap_index = pl.read(partition_indices, [0, partition_left])
                                            moving_value = pl.read(partition_values, [0, partition_right])
                                            moving_index = pl.read(partition_indices, [0, partition_right])
                                            pl.write(partition_values, [0, partition_left], moving_value)
                                            pl.write(partition_indices, [0, partition_left], moving_index)
                                            pl.write(partition_values, [0, partition_right], swap_value)
                                            pl.write(partition_indices, [0, partition_right], swap_index)
                                            partition_left = partition_left + 1
                                if partition_left <= 7:
                                    first = partition_left
                                else:
                                    last = partition_left
                    for insertion_column in pl.range(first + 1, last):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_19 in pl.range(cols + 8):
                            if insertion_hole > first:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    for insertion_column in pl.range(0 + 1, 7):
                        insertion_value = pl.read(partition_values, [0, insertion_column])
                        insertion_index = pl.read(partition_indices, [0, insertion_column])
                        insertion_hole = insertion_column
                        for bounded_step_20 in pl.range(cols + 8):
                            if insertion_hole > 0:
                                previous_value = pl.read(partition_values, [0, insertion_hole - 1])
                                if not insertion_value <= previous_value:
                                    moving_value = pl.read(partition_values, [0, insertion_hole - 1])
                                    moving_index = pl.read(partition_indices, [0, insertion_hole - 1])
                                    pl.write(partition_values, [0, insertion_hole], moving_value)
                                    pl.write(partition_indices, [0, insertion_hole], moving_index)
                                    insertion_hole = insertion_hole - 1
                        pl.write(partition_values, [0, insertion_hole], insertion_value)
                        pl.write(partition_indices, [0, insertion_hole], insertion_index)
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    for slot in pl.range(8):
                        copied_value = pl.read(partition_values, [0, slot])
                        copied_index = pl.read(partition_indices, [0, slot])
                        pl.write(selected_values, [0, slot], copied_value)
                        pl.write(selected_indices, [0, slot], copied_index)
                else:
                    heap_value_0 = pl.read(probabilities, [0, 0])
                    heap_index_0 = pl.cast(0, pl.INT32)
                    heap_value_1 = pl.read(probabilities, [0, 1])
                    heap_index_1 = pl.cast(1, pl.INT32)
                    heap_value_2 = pl.read(probabilities, [0, 2])
                    heap_index_2 = pl.cast(2, pl.INT32)
                    heap_value_3 = pl.read(probabilities, [0, 3])
                    heap_index_3 = pl.cast(3, pl.INT32)
                    heap_value_4 = pl.read(probabilities, [0, 4])
                    heap_index_4 = pl.cast(4, pl.INT32)
                    heap_value_5 = pl.read(probabilities, [0, 5])
                    heap_index_5 = pl.cast(5, pl.INT32)
                    heap_value_6 = pl.read(probabilities, [0, 6])
                    heap_index_6 = pl.cast(6, pl.INT32)
                    heap_value_7 = pl.read(probabilities, [0, 7])
                    heap_index_7 = pl.cast(7, pl.INT32)
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_7
                    heap_index_3 = heap_index_7
                    if heap_value_3 > incoming_value:
                        heap_value_7 = heap_value_3
                        heap_index_7 = heap_index_3
                        heap_value_3 = incoming_value
                        heap_index_3 = incoming_index
                    else:
                        heap_value_7 = incoming_value
                        heap_index_7 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    if heap_value_6 > heap_value_5:
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    else:
                        heap_value_2 = heap_value_6
                        heap_index_2 = heap_index_6
                        if heap_value_2 > incoming_value:
                            heap_value_6 = heap_value_2
                            heap_index_6 = heap_index_2
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                        else:
                            heap_value_6 = incoming_value
                            heap_index_6 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    if heap_value_4 > heap_value_3:
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        heap_value_3 = heap_value_7
                        heap_index_3 = heap_index_7
                        if heap_value_3 > incoming_value:
                            heap_value_7 = heap_value_3
                            heap_index_7 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_7 = incoming_value
                            heap_index_7 = incoming_index
                    else:
                        heap_value_1 = heap_value_4
                        heap_index_1 = heap_index_4
                        if heap_value_1 > incoming_value:
                            heap_value_4 = heap_value_1
                            heap_index_4 = heap_index_1
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                        else:
                            heap_value_4 = incoming_value
                            heap_index_4 = incoming_index
                    incoming_value = heap_value_0
                    incoming_index = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            heap_value_3 = heap_value_7
                            heap_index_3 = heap_index_7
                            if heap_value_3 > incoming_value:
                                heap_value_7 = heap_value_3
                                heap_index_7 = heap_index_3
                                if heap_value_1 > incoming_value:
                                    heap_value_3 = heap_value_1
                                    heap_index_3 = heap_index_1
                                    if heap_value_0 > incoming_value:
                                        heap_value_1 = heap_value_0
                                        heap_index_1 = heap_index_0
                                        heap_value_0 = incoming_value
                                        heap_index_0 = incoming_index
                                    else:
                                        heap_value_1 = incoming_value
                                        heap_index_1 = incoming_index
                                else:
                                    heap_value_3 = incoming_value
                                    heap_index_3 = incoming_index
                            else:
                                heap_value_7 = incoming_value
                                heap_index_7 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    summary_prefix = pl.slice(probabilities, [1, RESIDENT_SUMMARY_PREFIX_WIDTH], [0, 0])
                    summary_values = pl.reshape(summary_prefix, [RESIDENT_SUMMARY_PREFIX_WIDTH // RESIDENT_BLOCK_WIDTH, RESIDENT_BLOCK_WIDTH])
                    block_maxima = pl.row_max(summary_values)
                    for block in pl.range(cols // RESIDENT_BLOCK_WIDTH):
                        column = block * RESIDENT_BLOCK_WIDTH
                        block_probabilities = pl.slice(probabilities, [1, RESIDENT_BLOCK_WIDTH], [0, column])
                        if block < RESIDENT_SUMMARY_BLOCKS:
                            block_max = pl.read(block_maxima, [block, 0])
                        else:
                            block_max_lanes = pl.col_max(pl.reshape(block_probabilities, [RESIDENT_BLOCK_WIDTH // 8, 8]))
                            block_max = pl.read(block_max_lanes, [0, 0])
                            for lane in pl.unroll(1, 8):
                                lane_value = pl.read(block_max_lanes, [0, lane])
                                if lane_value > block_max:
                                    block_max = lane_value
                        if block_max > heap_value_0:
                            for block_offset in pl.range(RESIDENT_BLOCK_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        path_first = 2
                                        path_first_value = heap_value_2
                                        path_first_index = heap_index_2
                                        path_second = 6
                                        path_second_value = heap_value_6
                                        path_second_index = heap_index_6
                                        if heap_value_2 > heap_value_1:
                                            path_first = 1
                                            path_first_value = heap_value_1
                                            path_first_index = heap_index_1
                                            if heap_value_4 > heap_value_3:
                                                path_second = 3
                                                path_second_value = heap_value_3
                                                path_second_index = heap_index_3
                                            else:
                                                path_second = 4
                                                path_second_value = heap_value_4
                                                path_second_index = heap_index_4
                                        else:
                                            path_first = 2
                                            path_first_value = heap_value_2
                                            path_first_index = heap_index_2
                                            if heap_value_6 > heap_value_5:
                                                path_second = 5
                                                path_second_value = heap_value_5
                                                path_second_index = heap_index_5
                                            else:
                                                path_second = 6
                                                path_second_value = heap_value_6
                                                path_second_index = heap_index_6
                                        heap_value_0 = path_first_value
                                        heap_index_0 = path_first_index
                                        if path_first == 1:
                                            heap_value_1 = path_second_value
                                            heap_index_1 = path_second_index
                                        else:
                                            heap_value_2 = path_second_value
                                            heap_index_2 = path_second_index
                                        path_hole = path_second
                                        path_active = True
                                        if path_second == 3:
                                            path_third_value = heap_value_7
                                            path_third_index = heap_index_7
                                            heap_value_3 = path_third_value
                                            heap_index_3 = path_third_index
                                            path_hole = 7
                                            if path_third_value > incoming_value:
                                                heap_value_7 = path_third_value
                                                heap_index_7 = path_third_index
                                                path_hole = 3
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_second_value > incoming_value:
                                                if path_hole == 3:
                                                    heap_value_3 = path_second_value
                                                    heap_index_3 = path_second_index
                                                elif path_hole == 4:
                                                    heap_value_4 = path_second_value
                                                    heap_index_4 = path_second_index
                                                elif path_hole == 5:
                                                    heap_value_5 = path_second_value
                                                    heap_index_5 = path_second_index
                                                elif path_hole == 6:
                                                    heap_value_6 = path_second_value
                                                    heap_index_6 = path_second_index
                                                path_hole = path_first
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_first_value > incoming_value:
                                                if path_hole == 1:
                                                    heap_value_1 = path_first_value
                                                    heap_index_1 = path_first_index
                                                elif path_hole == 2:
                                                    heap_value_2 = path_first_value
                                                    heap_index_2 = path_first_index
                                                path_hole = 0
                                        if path_hole == 0:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        elif path_hole == 1:
                                            heap_value_1 = incoming_value
                                            heap_index_1 = incoming_index
                                        elif path_hole == 2:
                                            heap_value_2 = incoming_value
                                            heap_index_2 = incoming_index
                                        elif path_hole == 3:
                                            heap_value_3 = incoming_value
                                            heap_index_3 = incoming_index
                                        elif path_hole == 4:
                                            heap_value_4 = incoming_value
                                            heap_index_4 = incoming_index
                                        elif path_hole == 5:
                                            heap_value_5 = incoming_value
                                            heap_index_5 = incoming_index
                                        elif path_hole == 6:
                                            heap_value_6 = incoming_value
                                            heap_index_6 = incoming_index
                                        elif path_hole == 7:
                                            heap_value_7 = incoming_value
                                            heap_index_7 = incoming_index
                    if cols % RESIDENT_BLOCK_WIDTH != 0:
                        column = cols // RESIDENT_BLOCK_WIDTH * RESIDENT_BLOCK_WIDTH
                        block_probabilities = pl.slice(probabilities, [1, RESIDENT_TAIL_WIDTH], [0, column])
                        tail_max_lanes = pl.col_max(pl.reshape(block_probabilities, [RESIDENT_TAIL_WIDTH // 8, 8]))
                        tail_max = pl.read(tail_max_lanes, [0, 0])
                        for lane in pl.unroll(1, 8):
                            lane_value = pl.read(tail_max_lanes, [0, lane])
                            if lane_value > tail_max:
                                tail_max = lane_value
                        if tail_max > heap_value_0:
                            for block_offset in pl.range(RESIDENT_TAIL_WIDTH):
                                candidate_column = column + block_offset
                                if candidate_column >= 8:
                                    incoming_value = pl.read(block_probabilities, [0, block_offset])
                                    if incoming_value > heap_value_0:
                                        incoming_index = pl.cast(candidate_column, pl.INT32)
                                        path_first = 2
                                        path_first_value = heap_value_2
                                        path_first_index = heap_index_2
                                        path_second = 6
                                        path_second_value = heap_value_6
                                        path_second_index = heap_index_6
                                        if heap_value_2 > heap_value_1:
                                            path_first = 1
                                            path_first_value = heap_value_1
                                            path_first_index = heap_index_1
                                            if heap_value_4 > heap_value_3:
                                                path_second = 3
                                                path_second_value = heap_value_3
                                                path_second_index = heap_index_3
                                            else:
                                                path_second = 4
                                                path_second_value = heap_value_4
                                                path_second_index = heap_index_4
                                        else:
                                            path_first = 2
                                            path_first_value = heap_value_2
                                            path_first_index = heap_index_2
                                            if heap_value_6 > heap_value_5:
                                                path_second = 5
                                                path_second_value = heap_value_5
                                                path_second_index = heap_index_5
                                            else:
                                                path_second = 6
                                                path_second_value = heap_value_6
                                                path_second_index = heap_index_6
                                        heap_value_0 = path_first_value
                                        heap_index_0 = path_first_index
                                        if path_first == 1:
                                            heap_value_1 = path_second_value
                                            heap_index_1 = path_second_index
                                        else:
                                            heap_value_2 = path_second_value
                                            heap_index_2 = path_second_index
                                        path_hole = path_second
                                        path_active = True
                                        if path_second == 3:
                                            path_third_value = heap_value_7
                                            path_third_index = heap_index_7
                                            heap_value_3 = path_third_value
                                            heap_index_3 = path_third_index
                                            path_hole = 7
                                            if path_third_value > incoming_value:
                                                heap_value_7 = path_third_value
                                                heap_index_7 = path_third_index
                                                path_hole = 3
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_second_value > incoming_value:
                                                if path_hole == 3:
                                                    heap_value_3 = path_second_value
                                                    heap_index_3 = path_second_index
                                                elif path_hole == 4:
                                                    heap_value_4 = path_second_value
                                                    heap_index_4 = path_second_index
                                                elif path_hole == 5:
                                                    heap_value_5 = path_second_value
                                                    heap_index_5 = path_second_index
                                                elif path_hole == 6:
                                                    heap_value_6 = path_second_value
                                                    heap_index_6 = path_second_index
                                                path_hole = path_first
                                            else:
                                                path_active = False
                                        if path_active:
                                            if path_first_value > incoming_value:
                                                if path_hole == 1:
                                                    heap_value_1 = path_first_value
                                                    heap_index_1 = path_first_index
                                                elif path_hole == 2:
                                                    heap_value_2 = path_first_value
                                                    heap_index_2 = path_first_index
                                                path_hole = 0
                                        if path_hole == 0:
                                            heap_value_0 = incoming_value
                                            heap_index_0 = incoming_index
                                        elif path_hole == 1:
                                            heap_value_1 = incoming_value
                                            heap_index_1 = incoming_index
                                        elif path_hole == 2:
                                            heap_value_2 = incoming_value
                                            heap_index_2 = incoming_index
                                        elif path_hole == 3:
                                            heap_value_3 = incoming_value
                                            heap_index_3 = incoming_index
                                        elif path_hole == 4:
                                            heap_value_4 = incoming_value
                                            heap_index_4 = incoming_index
                                        elif path_hole == 5:
                                            heap_value_5 = incoming_value
                                            heap_index_5 = incoming_index
                                        elif path_hole == 6:
                                            heap_value_6 = incoming_value
                                            heap_index_6 = incoming_index
                                        elif path_hole == 7:
                                            heap_value_7 = incoming_value
                                            heap_index_7 = incoming_index
                    incoming_value = heap_value_7
                    incoming_index = heap_index_7
                    heap_value_7 = heap_value_0
                    heap_index_7 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_6 > heap_value_5:
                            heap_value_2 = heap_value_5
                            heap_index_2 = heap_index_5
                            if heap_value_2 > incoming_value:
                                heap_value_5 = heap_value_2
                                heap_index_5 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_5 = incoming_value
                                heap_index_5 = incoming_index
                        else:
                            heap_value_2 = heap_value_6
                            heap_index_2 = heap_index_6
                            if heap_value_2 > incoming_value:
                                heap_value_6 = heap_value_2
                                heap_index_6 = heap_index_2
                                if heap_value_0 > incoming_value:
                                    heap_value_2 = heap_value_0
                                    heap_index_2 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_2 = incoming_value
                                    heap_index_2 = incoming_index
                            else:
                                heap_value_6 = incoming_value
                                heap_index_6 = incoming_index
                    incoming_value = heap_value_6
                    incoming_index = heap_index_6
                    heap_value_6 = heap_value_0
                    heap_index_6 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        heap_value_2 = heap_value_5
                        heap_index_2 = heap_index_5
                        if heap_value_2 > incoming_value:
                            heap_value_5 = heap_value_2
                            heap_index_5 = heap_index_2
                            if heap_value_0 > incoming_value:
                                heap_value_2 = heap_value_0
                                heap_index_2 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_2 = incoming_value
                                heap_index_2 = incoming_index
                        else:
                            heap_value_5 = incoming_value
                            heap_index_5 = incoming_index
                    incoming_value = heap_value_5
                    incoming_index = heap_index_5
                    heap_value_5 = heap_value_0
                    heap_index_5 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_4 > heap_value_3:
                            heap_value_1 = heap_value_3
                            heap_index_1 = heap_index_3
                            if heap_value_1 > incoming_value:
                                heap_value_3 = heap_value_1
                                heap_index_3 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_3 = incoming_value
                                heap_index_3 = incoming_index
                        else:
                            heap_value_1 = heap_value_4
                            heap_index_1 = heap_index_4
                            if heap_value_1 > incoming_value:
                                heap_value_4 = heap_value_1
                                heap_index_4 = heap_index_1
                                if heap_value_0 > incoming_value:
                                    heap_value_1 = heap_value_0
                                    heap_index_1 = heap_index_0
                                    heap_value_0 = incoming_value
                                    heap_index_0 = incoming_index
                                else:
                                    heap_value_1 = incoming_value
                                    heap_index_1 = incoming_index
                            else:
                                heap_value_4 = incoming_value
                                heap_index_4 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_4
                    incoming_index = heap_index_4
                    heap_value_4 = heap_value_0
                    heap_index_4 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        heap_value_1 = heap_value_3
                        heap_index_1 = heap_index_3
                        if heap_value_1 > incoming_value:
                            heap_value_3 = heap_value_1
                            heap_index_3 = heap_index_1
                            if heap_value_0 > incoming_value:
                                heap_value_1 = heap_value_0
                                heap_index_1 = heap_index_0
                                heap_value_0 = incoming_value
                                heap_index_0 = incoming_index
                            else:
                                heap_value_1 = incoming_value
                                heap_index_1 = incoming_index
                        else:
                            heap_value_3 = incoming_value
                            heap_index_3 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_3
                    incoming_index = heap_index_3
                    heap_value_3 = heap_value_0
                    heap_index_3 = heap_index_0
                    if heap_value_2 > heap_value_1:
                        heap_value_0 = heap_value_1
                        heap_index_0 = heap_index_1
                        if heap_value_0 > incoming_value:
                            heap_value_1 = heap_value_0
                            heap_index_1 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_1 = incoming_value
                            heap_index_1 = incoming_index
                    else:
                        heap_value_0 = heap_value_2
                        heap_index_0 = heap_index_2
                        if heap_value_0 > incoming_value:
                            heap_value_2 = heap_value_0
                            heap_index_2 = heap_index_0
                            heap_value_0 = incoming_value
                            heap_index_0 = incoming_index
                        else:
                            heap_value_2 = incoming_value
                            heap_index_2 = incoming_index
                    incoming_value = heap_value_2
                    incoming_index = heap_index_2
                    heap_value_2 = heap_value_0
                    heap_index_2 = heap_index_0
                    heap_value_0 = heap_value_1
                    heap_index_0 = heap_index_1
                    if heap_value_0 > incoming_value:
                        heap_value_1 = heap_value_0
                        heap_index_1 = heap_index_0
                        heap_value_0 = incoming_value
                        heap_index_0 = incoming_index
                    else:
                        heap_value_1 = incoming_value
                        heap_index_1 = incoming_index
                    incoming_value = heap_value_1
                    incoming_index = heap_index_1
                    heap_value_1 = heap_value_0
                    heap_index_1 = heap_index_0
                    heap_value_0 = incoming_value
                    heap_index_0 = incoming_index
                    selected_values = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                    selected_indices = pl.full([1, 8], dtype=pl.INT32, value=0)
                    pl.write(selected_values, [0, 0], heap_value_0)
                    pl.write(selected_indices, [0, 0], heap_index_0)
                    pl.write(selected_values, [0, 1], heap_value_1)
                    pl.write(selected_indices, [0, 1], heap_index_1)
                    pl.write(selected_values, [0, 2], heap_value_2)
                    pl.write(selected_indices, [0, 2], heap_index_2)
                    pl.write(selected_values, [0, 3], heap_value_3)
                    pl.write(selected_indices, [0, 3], heap_index_3)
                    pl.write(selected_values, [0, 4], heap_value_4)
                    pl.write(selected_indices, [0, 4], heap_index_4)
                    pl.write(selected_values, [0, 5], heap_value_5)
                    pl.write(selected_indices, [0, 5], heap_index_5)
                    pl.write(selected_values, [0, 6], heap_value_6)
                    pl.write(selected_indices, [0, 6], heap_index_6)
                    pl.write(selected_values, [0, 7], heap_value_7)
                    pl.write(selected_indices, [0, 7], heap_index_7)
                selected_sum = pl.read(selected_values, [0, 0])
                for slot in pl.unroll(1, 8):
                    selected_sum = selected_sum + pl.read(selected_values, [0, slot])
                weights = pl.div(selected_values, selected_sum + 1e-08)
                for slot in pl.unroll(8):
                    winner = pl.cast(pl.read(selected_indices, [0, slot]), pl.INDEX)
                    original_probability = pl.read(probabilities, [0, winner])
                    pl.write(probabilities, [0, winner], original_probability - 2.0)
                p = pl.assemble(p, probabilities, [row, 0])
                topk_w = pl.assemble(topk_w, weights, [row, 0])
                topk_idx = pl.assemble(topk_idx, selected_indices, [row, 0])
    return (topk_w, topk_idx, p)


def golden_gating(tensors):
    import torch
    logits = tensors['logits'].float()
    exponentials = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    values, indices = torch.topk(probabilities, 8, dim=-1)
    tensors['topk_w'][:] = values / (values.sum(dim=-1, keepdim=True) + 1e-08)
    tensors['topk_idx'][:] = indices.to(torch.int32)
    tensors['p'][:] = probabilities.scatter(1, indices, probabilities.gather(1, indices) - 2.0)

def build(shape):
    import torch
    from golden import TensorSpec
    global SUMMARY_RESIDUAL_WIDTH, SUMMARY_RESIDUAL_OFFSET
    global NARROW_DEPTH, RESIDENT_BLOCK_WIDTH, RESIDENT_TAIL_WIDTH
    global RESIDENT_SUMMARY_BLOCKS, RESIDENT_SUMMARY_PREFIX_WIDTH
    global PARENT_SUMMARY_PREFIX_WIDTH, SUFFIX_SUMMARY_PREFIX_WIDTH, SUFFIX_SUMMARY_TILE_WIDTH, SUFFIX_SUMMARY_OFFSET
    rows, cols = (int(shape['M']), int(shape['N']))
    dtype = shape.get('dtype', 'fp32')
    summary_prefix = cols // (RESIDENT_BLOCK_WIDTH * 8) * (RESIDENT_BLOCK_WIDTH * 8)
    SUMMARY_RESIDUAL_WIDTH = max(128, cols - summary_prefix)
    SUMMARY_RESIDUAL_OFFSET = summary_prefix if cols > summary_prefix else 0
    if rows < 1 or cols < 128 or cols % 128 or (dtype not in ('fp32', 'bf16')):
        raise ValueError('Expected positive M, N divisible by 128, and fp32 or bf16 logits')
    NARROW_DEPTH = 2 * (cols.bit_length() - 1)
    RESIDENT_BLOCK_WIDTH = min(RESIDENT_REQUESTED_BLOCK_WIDTH, cols // 8)
    RESIDENT_TAIL_WIDTH = cols % RESIDENT_BLOCK_WIDTH or 128
    RESIDENT_SUMMARY_BLOCKS = max(8, cols // RESIDENT_BLOCK_WIDTH // 8 * 8)
    RESIDENT_SUMMARY_PREFIX_WIDTH = min(cols // RESIDENT_BLOCK_WIDTH, RESIDENT_SUMMARY_BLOCKS) * RESIDENT_BLOCK_WIDTH
    PARENT_SUMMARY_PREFIX_WIDTH = cols // PARENT_WIDTH // 8 * 8 * PARENT_WIDTH
    SUFFIX_SUMMARY_PREFIX_WIDTH = (cols - PARENT_SUMMARY_PREFIX_WIDTH) // RESIDENT_BLOCK_WIDTH // 8 * 8 * RESIDENT_BLOCK_WIDTH
    SUFFIX_SUMMARY_TILE_WIDTH = max(256, SUFFIX_SUMMARY_PREFIX_WIDTH)
    SUFFIX_SUMMARY_OFFSET = PARENT_SUMMARY_PREFIX_WIDTH if SUFFIX_SUMMARY_PREFIX_WIDTH else 0
    specs = [TensorSpec('logits', [rows, cols], torch.bfloat16 if dtype == 'bf16' else torch.float32, init_value=torch.randn), TensorSpec('colidx', [1, cols], torch.float32, init_value=lambda: torch.arange(cols, dtype=torch.float32).reshape(1, cols)), TensorSpec('p', [rows, cols], torch.float32), TensorSpec('topk_w', [rows, 8], torch.float32), TensorSpec('topk_idx', [rows, 8], torch.int32)]
    if dtype == 'bf16':
        fn = (flat_resident if cols < PARENT_WIDTH * 8 else resident) if cols <= RESIDENT_CAPACITY else bounded
    else:
        fn = (flat_resident_fp32 if cols < PARENT_WIDTH * 8 else resident_fp32) if cols <= RESIDENT_CAPACITY else bounded_fp32
    return (fn, specs, golden_gating, 0.0001, 0.0001)

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
