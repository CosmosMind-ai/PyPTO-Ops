# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact grouped routing with BF16 storage and FP32 selection state."""
from __future__ import annotations
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}

RTOL = 0.0001
ATOL = 0.0001
G = 8
TG = 4
K = 8
SCALE = 2.5
NEG = 1e+30
GT = 16
INPUT_DTYPE = pl.FP32
M = 512
N = 1024
EPG = 128
ROW_TILE = 64
COL_TILE = 128
GROUP_ELEMENT_CAP = 8192
GROUP_WIDTH = 128
MASK_WIDTH_CAP = 128
PRODUCER_ELEMENT_CAP = 8192
EXTRACT_WIDTH_CAP = 256
EXTRACTION_ROWS = 16
EXTRACTION_ELEMENT_CAP = 8192
EXTRACTION_BLOCKS = 8
MASK_WIDTH = 16
EXTRACT_WIDTH = 128
RESIDENT_GROUP = True
FRONTIER_ROWS = 512
LOCAL_FRONTIER = False

@pl.jit
def grouped_topk_fp32(logits: pl.Tensor[[M, N], pl.FP32], bias: pl.Tensor[[1, N], pl.FP32], colidx: pl.Tensor[[1, N], pl.FP32], s: pl.InOut[pl.Tensor[[M, N], pl.FP32]], sel: pl.InOut[pl.Tensor[[M, N], pl.FP32]], smasked: pl.InOut[pl.Tensor[[M, N], pl.FP32]], gscore_t: pl.InOut[pl.Tensor[[G, M], pl.FP32]], gmask_t: pl.InOut[pl.Tensor[[G, M], pl.FP32]], topk_w_t: pl.InOut[pl.Tensor[[K, M], pl.FP32]], topk_idx_t: pl.InOut[pl.Tensor[[K, M], pl.INT32]], selsum: pl.InOut[pl.Tensor[[1, M], pl.FP32]], topk_w: pl.Out[pl.Tensor[[M, K], pl.FP32]], topk_idx: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
    maximum_frontier = pl.create_tensor([1, FRONTIER_ROWS], dtype=pl.FP32)
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='sigmoid'):
            for cb in pl.range(N // COL_TILE):
                c0 = cb * COL_TILE
                xt = pl.slice(logits, [ROW_TILE, COL_TILE], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
                s[r:r + ROW_TILE, c0:c0 + COL_TILE] = pl.slice(pl.div(pl.slice(pl.full([ROW_TILE, COL_TILE], dtype=pl.FP32, value=1.0), [ROW_TILE, COL_TILE], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE]), pl.add(pl.exp(pl.neg(xt)), 1.0)), [ROW_TILE, COL_TILE], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='bias'):
            for cb in pl.range(N // COL_TILE):
                c0 = cb * COL_TILE
                st = pl.slice(s, [ROW_TILE, COL_TILE], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
                bt = bias[0:1, c0:c0 + COL_TILE]
                sel[r:r + ROW_TILE, c0:c0 + COL_TILE] = pl.slice(pl.col_expand_add(st, bt), [ROW_TILE, COL_TILE], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
    for g in pl.range(G):
        g0 = g * EPG
        if RESIDENT_GROUP:
            for r in pl.parallel(0, M, ROW_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='group_score_resident'):
                    group_tile = pl.slice(sel, [ROW_TILE, EPG], [r, g0], valid_shape=[pl.min(ROW_TILE, M - r), EPG])
                    first_max = pl.row_max(group_tile)
                    first_winners = pl.cmp(pl.row_expand_sub(group_tile, first_max), 0.0, cmp_type=0)
                    remaining = pl.sub(group_tile, pl.mul(first_winners, NEG))
                    second_max = pl.row_max(remaining)
                    group_score = pl.reshape(pl.add(first_max, second_max), [1, ROW_TILE])
                    gscore_t[g:g + 1, r:r + ROW_TILE] = pl.slice(group_score, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
        else:
            for r in pl.parallel(0, M, ROW_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='group_first_stream'):
                    first_acc = pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=-1e+38), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
                    for cb in pl.range(EPG // GROUP_WIDTH):
                        c0 = g0 + cb * GROUP_WIDTH
                        scores = pl.slice(sel, [ROW_TILE, GROUP_WIDTH], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), GROUP_WIDTH])
                        first_acc = pl.maximum(first_acc, pl.reshape(pl.row_max(scores), [1, ROW_TILE]))
                    gscore_t[g:g + 1, r:r + ROW_TILE] = pl.slice(first_acc, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            for r in pl.parallel(0, M, ROW_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='group_second_stream'):
                    stream_first_max = pl.reshape(pl.slice(gscore_t, [1, ROW_TILE], [g, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]), [ROW_TILE, 1])
                    second_acc = pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=-1e+38), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
                    for cb in pl.range(EPG // GROUP_WIDTH):
                        c0 = g0 + cb * GROUP_WIDTH
                        scores = pl.slice(sel, [ROW_TILE, GROUP_WIDTH], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), GROUP_WIDTH])
                        stream_winners = pl.cmp(pl.row_expand_sub(scores, stream_first_max), 0.0, cmp_type=0)
                        stream_remaining = pl.sub(scores, pl.mul(stream_winners, NEG))
                        second_acc = pl.maximum(second_acc, pl.reshape(pl.row_max(stream_remaining), [1, ROW_TILE]))
                    gscore_t[g:g + 1, r:r + ROW_TILE] = pl.slice(pl.add(pl.reshape(stream_first_max, [1, ROW_TILE]), second_acc), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='gzero'):
            gmask_t[0:G, r:r + ROW_TILE] = pl.slice(pl.slice(pl.full([G, ROW_TILE], dtype=pl.FP32, value=0.0), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)]), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
    for kb in pl.range(TG):
        for r in pl.parallel(0, M, ROW_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='gsel'):
                gt = pl.slice(gscore_t, [G, ROW_TILE], [0, r], valid_shape=[G, pl.min(ROW_TILE, M - r)])
                mx = pl.slice(gscore_t, [1, ROW_TILE], [0, r], valid_shape=[1, pl.min(ROW_TILE, M - r)])
                for g2 in pl.range(1, G):
                    mx = pl.maximum(mx, pl.slice(gscore_t, [1, ROW_TILE], [g2, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]))
                zeros8 = pl.slice(pl.full([G, ROW_TILE], dtype=pl.FP32, value=0.0), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
                mx_e = pl.col_expand(zeros8, mx)
                onehot8 = pl.cmp(gt, mx_e, cmp_type=0)
                gmask_t[0:G, r:r + ROW_TILE] = pl.slice(pl.add(pl.slice(gmask_t, [G, ROW_TILE], [0, r], valid_shape=[G, pl.min(ROW_TILE, M - r)]), onehot8), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
                gscore_t[0:G, r:r + ROW_TILE] = pl.slice(pl.sub(gt, pl.mul(onehot8, NEG)), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='mask_producer_seed'):
            producer_max = pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            for g in pl.range(G):
                g0 = g * EPG
                producer_mask = pl.reshape(pl.slice(gmask_t, [1, ROW_TILE], [g, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]), [ROW_TILE, 1])
                producer_ones = pl.slice(pl.full([ROW_TILE, MASK_WIDTH], dtype=pl.FP32, value=1.0), [ROW_TILE, MASK_WIDTH], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), MASK_WIDTH])
                producer_keep = pl.row_expand_mul(producer_ones, producer_mask)
                for cb in pl.range(EPG // MASK_WIDTH):
                    c0 = g0 + cb * MASK_WIDTH
                    producer_scores = pl.slice(sel, [ROW_TILE, MASK_WIDTH], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), MASK_WIDTH])
                    masked_tile = pl.sub(pl.row_expand_mul(producer_scores, producer_mask), pl.mul(pl.sub(producer_ones, producer_keep), NEG))
                    smasked[r:r + ROW_TILE, c0:c0 + MASK_WIDTH] = pl.slice(masked_tile, [ROW_TILE, MASK_WIDTH], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), MASK_WIDTH])
                    producer_tile_max = pl.reshape(pl.row_max(masked_tile), [1, ROW_TILE])
                    if g == 0 and cb == 0:
                        producer_max = producer_tile_max
                    else:
                        producer_max = pl.maximum(producer_max, producer_tile_max)
            maximum_frontier[0:1, r:r + ROW_TILE] = pl.slice(producer_max, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
    for owner in pl.spmd(EXTRACTION_BLOCKS, name_hint='spmd_owned_extract'):
        r = owner * EXTRACTION_ROWS
        current_frontier = pl.slice(maximum_frontier, [1, EXTRACTION_ROWS], [0, r], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
        for kb in pl.range(K):
            current_max = pl.reshape(current_frontier, [EXTRACTION_ROWS, 1])
            index_sum = pl.slice(pl.full([1, EXTRACTION_ROWS], dtype=pl.FP32, value=0.0), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            weight_sum = pl.slice(pl.full([1, EXTRACTION_ROWS], dtype=pl.FP32, value=0.0), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            next_max = pl.slice(pl.full([1, EXTRACTION_ROWS], dtype=pl.FP32, value=0.0), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            for cb in pl.range(N // EXTRACT_WIDTH):
                c0 = cb * EXTRACT_WIDTH
                expert_scores = pl.slice(smasked, [EXTRACTION_ROWS, EXTRACT_WIDTH], [r, c0], valid_shape=[pl.min(EXTRACTION_ROWS, M - r), EXTRACT_WIDTH])
                winners = pl.cmp(pl.row_expand_sub(expert_scores, current_max), 0.0, cmp_type=0)
                selected_indices = pl.col_expand_mul(winners, colidx[0:1, c0:c0 + EXTRACT_WIDTH])
                index_sum = pl.add(index_sum, pl.reshape(pl.row_sum(selected_indices), [1, EXTRACTION_ROWS]))
                bare_scores = pl.slice(s, [EXTRACTION_ROWS, EXTRACT_WIDTH], [r, c0], valid_shape=[pl.min(EXTRACTION_ROWS, M - r), EXTRACT_WIDTH])
                weight_sum = pl.add(weight_sum, pl.reshape(pl.row_sum(pl.mul(bare_scores, winners)), [1, EXTRACTION_ROWS]))
                updated = pl.sub(expert_scores, pl.mul(winners, NEG))
                smasked[r:r + EXTRACTION_ROWS, c0:c0 + EXTRACT_WIDTH] = pl.slice(updated, [EXTRACTION_ROWS, EXTRACT_WIDTH], [0, 0], valid_shape=[pl.min(EXTRACTION_ROWS, M - r), EXTRACT_WIDTH])
                if kb < K - 1:
                    tile_max = pl.reshape(pl.row_max(updated), [1, EXTRACTION_ROWS])
                    if cb == 0:
                        next_max = tile_max
                    else:
                        next_max = pl.maximum(next_max, tile_max)
            topk_w_t[kb:kb + 1, r:r + EXTRACTION_ROWS] = pl.slice(weight_sum, [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            topk_idx_t[kb:kb + 1, r:r + EXTRACTION_ROWS] = pl.slice(pl.cast(index_sum, target_type=pl.INT32, mode='rint'), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            if kb < K - 1:
                current_frontier = next_max
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='renorm'):
            ss = pl.slice(topk_w_t, [1, ROW_TILE], [0, r], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            for kb in pl.range(1, K):
                ss = pl.add(ss, pl.slice(topk_w_t, [1, ROW_TILE], [kb, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]))
            selsum[0:1, r:r + ROW_TILE] = pl.slice(ss, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            inv = pl.div(pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=SCALE), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)]), pl.add(ss, 1e-20))
            wt_t = pl.slice(topk_w_t, [K, ROW_TILE], [0, r], valid_shape=[K, pl.min(ROW_TILE, M - r)])
            topk_w_t[0:K, r:r + ROW_TILE] = pl.slice(pl.col_expand_mul(wt_t, inv), [K, ROW_TILE], [0, 0], valid_shape=[K, pl.min(ROW_TILE, M - r)])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='owned_vector_weights'):
            export_weights = pl.slice(topk_w_t, [K, ROW_TILE], [0, r], valid_shape=[K, pl.min(ROW_TILE, M - r)])
            padded_weights = pl.full([16, ROW_TILE], dtype=pl.FP32, value=0.0)
            padded_weights = pl.assemble(padded_weights, export_weights, [0, 0])
            transposed_weights = pl.transpose(padded_weights, 0, 1)
            topk_w[r:r + ROW_TILE, 0:K] = pl.slice(transposed_weights, [ROW_TILE, K], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), K])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='owned_vector_indices'):
            export_indices = pl.slice(topk_idx_t, [K, ROW_TILE], [0, r], valid_shape=[K, pl.min(ROW_TILE, M - r)])
            padded_indices = pl.full([16, ROW_TILE], dtype=pl.INT32, value=0)
            padded_indices = pl.assemble(padded_indices, export_indices, [0, 0])
            transposed_indices = pl.transpose(padded_indices, 0, 1)
            topk_idx[r:r + ROW_TILE, 0:K] = pl.slice(transposed_indices, [ROW_TILE, K], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), K])
    return topk_w

@pl.jit
def grouped_topk_bf16(logits: pl.Tensor[[M, N], pl.BF16], bias: pl.Tensor[[1, N], pl.FP32], colidx: pl.Tensor[[1, N], pl.FP32], s: pl.InOut[pl.Tensor[[M, N], pl.FP32]], sel: pl.InOut[pl.Tensor[[M, N], pl.FP32]], smasked: pl.InOut[pl.Tensor[[M, N], pl.FP32]], gscore_t: pl.InOut[pl.Tensor[[G, M], pl.FP32]], gmask_t: pl.InOut[pl.Tensor[[G, M], pl.FP32]], topk_w_t: pl.InOut[pl.Tensor[[K, M], pl.FP32]], topk_idx_t: pl.InOut[pl.Tensor[[K, M], pl.INT32]], selsum: pl.InOut[pl.Tensor[[1, M], pl.FP32]], topk_w: pl.Out[pl.Tensor[[M, K], pl.FP32]], topk_idx: pl.Out[pl.Tensor[[M, K], pl.INT32]]):
    maximum_frontier = pl.create_tensor([1, FRONTIER_ROWS], dtype=pl.FP32)
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='sigmoid'):
            for cb in pl.range(N // COL_TILE):
                c0 = cb * COL_TILE
                xt = pl.cast(pl.slice(logits, [ROW_TILE, COL_TILE], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE]), target_type=pl.FP32)
                s[r:r + ROW_TILE, c0:c0 + COL_TILE] = pl.slice(pl.div(pl.slice(pl.full([ROW_TILE, COL_TILE], dtype=pl.FP32, value=1.0), [ROW_TILE, COL_TILE], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE]), pl.add(pl.exp(pl.neg(xt)), 1.0)), [ROW_TILE, COL_TILE], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='bias'):
            for cb in pl.range(N // COL_TILE):
                c0 = cb * COL_TILE
                st = pl.slice(s, [ROW_TILE, COL_TILE], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
                bt = bias[0:1, c0:c0 + COL_TILE]
                sel[r:r + ROW_TILE, c0:c0 + COL_TILE] = pl.slice(pl.col_expand_add(st, bt), [ROW_TILE, COL_TILE], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), COL_TILE])
    for g in pl.range(G):
        g0 = g * EPG
        if RESIDENT_GROUP:
            for r in pl.parallel(0, M, ROW_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='group_score_resident'):
                    group_tile = pl.slice(sel, [ROW_TILE, EPG], [r, g0], valid_shape=[pl.min(ROW_TILE, M - r), EPG])
                    first_max = pl.row_max(group_tile)
                    first_winners = pl.cmp(pl.row_expand_sub(group_tile, first_max), 0.0, cmp_type=0)
                    remaining = pl.sub(group_tile, pl.mul(first_winners, NEG))
                    second_max = pl.row_max(remaining)
                    group_score = pl.reshape(pl.add(first_max, second_max), [1, ROW_TILE])
                    gscore_t[g:g + 1, r:r + ROW_TILE] = pl.slice(group_score, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
        else:
            for r in pl.parallel(0, M, ROW_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='group_first_stream'):
                    first_acc = pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=-1e+38), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
                    for cb in pl.range(EPG // GROUP_WIDTH):
                        c0 = g0 + cb * GROUP_WIDTH
                        scores = pl.slice(sel, [ROW_TILE, GROUP_WIDTH], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), GROUP_WIDTH])
                        first_acc = pl.maximum(first_acc, pl.reshape(pl.row_max(scores), [1, ROW_TILE]))
                    gscore_t[g:g + 1, r:r + ROW_TILE] = pl.slice(first_acc, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            for r in pl.parallel(0, M, ROW_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='group_second_stream'):
                    stream_first_max = pl.reshape(pl.slice(gscore_t, [1, ROW_TILE], [g, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]), [ROW_TILE, 1])
                    second_acc = pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=-1e+38), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
                    for cb in pl.range(EPG // GROUP_WIDTH):
                        c0 = g0 + cb * GROUP_WIDTH
                        scores = pl.slice(sel, [ROW_TILE, GROUP_WIDTH], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), GROUP_WIDTH])
                        stream_winners = pl.cmp(pl.row_expand_sub(scores, stream_first_max), 0.0, cmp_type=0)
                        stream_remaining = pl.sub(scores, pl.mul(stream_winners, NEG))
                        second_acc = pl.maximum(second_acc, pl.reshape(pl.row_max(stream_remaining), [1, ROW_TILE]))
                    gscore_t[g:g + 1, r:r + ROW_TILE] = pl.slice(pl.add(pl.reshape(stream_first_max, [1, ROW_TILE]), second_acc), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='gzero'):
            gmask_t[0:G, r:r + ROW_TILE] = pl.slice(pl.slice(pl.full([G, ROW_TILE], dtype=pl.FP32, value=0.0), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)]), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
    for kb in pl.range(TG):
        for r in pl.parallel(0, M, ROW_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='gsel'):
                gt = pl.slice(gscore_t, [G, ROW_TILE], [0, r], valid_shape=[G, pl.min(ROW_TILE, M - r)])
                mx = pl.slice(gscore_t, [1, ROW_TILE], [0, r], valid_shape=[1, pl.min(ROW_TILE, M - r)])
                for g2 in pl.range(1, G):
                    mx = pl.maximum(mx, pl.slice(gscore_t, [1, ROW_TILE], [g2, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]))
                zeros8 = pl.slice(pl.full([G, ROW_TILE], dtype=pl.FP32, value=0.0), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
                mx_e = pl.col_expand(zeros8, mx)
                onehot8 = pl.cmp(gt, mx_e, cmp_type=0)
                gmask_t[0:G, r:r + ROW_TILE] = pl.slice(pl.add(pl.slice(gmask_t, [G, ROW_TILE], [0, r], valid_shape=[G, pl.min(ROW_TILE, M - r)]), onehot8), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
                gscore_t[0:G, r:r + ROW_TILE] = pl.slice(pl.sub(gt, pl.mul(onehot8, NEG)), [G, ROW_TILE], [0, 0], valid_shape=[G, pl.min(ROW_TILE, M - r)])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='mask_producer_seed'):
            producer_max = pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            for g in pl.range(G):
                g0 = g * EPG
                producer_mask = pl.reshape(pl.slice(gmask_t, [1, ROW_TILE], [g, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]), [ROW_TILE, 1])
                producer_ones = pl.slice(pl.full([ROW_TILE, MASK_WIDTH], dtype=pl.FP32, value=1.0), [ROW_TILE, MASK_WIDTH], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), MASK_WIDTH])
                producer_keep = pl.row_expand_mul(producer_ones, producer_mask)
                for cb in pl.range(EPG // MASK_WIDTH):
                    c0 = g0 + cb * MASK_WIDTH
                    producer_scores = pl.slice(sel, [ROW_TILE, MASK_WIDTH], [r, c0], valid_shape=[pl.min(ROW_TILE, M - r), MASK_WIDTH])
                    masked_tile = pl.sub(pl.row_expand_mul(producer_scores, producer_mask), pl.mul(pl.sub(producer_ones, producer_keep), NEG))
                    smasked[r:r + ROW_TILE, c0:c0 + MASK_WIDTH] = pl.slice(masked_tile, [ROW_TILE, MASK_WIDTH], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), MASK_WIDTH])
                    producer_tile_max = pl.reshape(pl.row_max(masked_tile), [1, ROW_TILE])
                    if g == 0 and cb == 0:
                        producer_max = producer_tile_max
                    else:
                        producer_max = pl.maximum(producer_max, producer_tile_max)
            maximum_frontier[0:1, r:r + ROW_TILE] = pl.slice(producer_max, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
    for owner in pl.spmd(EXTRACTION_BLOCKS, name_hint='spmd_owned_extract'):
        r = owner * EXTRACTION_ROWS
        current_frontier = pl.slice(maximum_frontier, [1, EXTRACTION_ROWS], [0, r], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
        for kb in pl.range(K):
            current_max = pl.reshape(current_frontier, [EXTRACTION_ROWS, 1])
            index_sum = pl.slice(pl.full([1, EXTRACTION_ROWS], dtype=pl.FP32, value=0.0), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            weight_sum = pl.slice(pl.full([1, EXTRACTION_ROWS], dtype=pl.FP32, value=0.0), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            next_max = pl.slice(pl.full([1, EXTRACTION_ROWS], dtype=pl.FP32, value=0.0), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            for cb in pl.range(N // EXTRACT_WIDTH):
                c0 = cb * EXTRACT_WIDTH
                expert_scores = pl.slice(smasked, [EXTRACTION_ROWS, EXTRACT_WIDTH], [r, c0], valid_shape=[pl.min(EXTRACTION_ROWS, M - r), EXTRACT_WIDTH])
                winners = pl.cmp(pl.row_expand_sub(expert_scores, current_max), 0.0, cmp_type=0)
                selected_indices = pl.col_expand_mul(winners, colidx[0:1, c0:c0 + EXTRACT_WIDTH])
                index_sum = pl.add(index_sum, pl.reshape(pl.row_sum(selected_indices), [1, EXTRACTION_ROWS]))
                bare_scores = pl.slice(s, [EXTRACTION_ROWS, EXTRACT_WIDTH], [r, c0], valid_shape=[pl.min(EXTRACTION_ROWS, M - r), EXTRACT_WIDTH])
                weight_sum = pl.add(weight_sum, pl.reshape(pl.row_sum(pl.mul(bare_scores, winners)), [1, EXTRACTION_ROWS]))
                updated = pl.sub(expert_scores, pl.mul(winners, NEG))
                smasked[r:r + EXTRACTION_ROWS, c0:c0 + EXTRACT_WIDTH] = pl.slice(updated, [EXTRACTION_ROWS, EXTRACT_WIDTH], [0, 0], valid_shape=[pl.min(EXTRACTION_ROWS, M - r), EXTRACT_WIDTH])
                if kb < K - 1:
                    tile_max = pl.reshape(pl.row_max(updated), [1, EXTRACTION_ROWS])
                    if cb == 0:
                        next_max = tile_max
                    else:
                        next_max = pl.maximum(next_max, tile_max)
            topk_w_t[kb:kb + 1, r:r + EXTRACTION_ROWS] = pl.slice(weight_sum, [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            topk_idx_t[kb:kb + 1, r:r + EXTRACTION_ROWS] = pl.slice(pl.cast(index_sum, target_type=pl.INT32, mode='rint'), [1, EXTRACTION_ROWS], [0, 0], valid_shape=[1, pl.min(EXTRACTION_ROWS, M - r)])
            if kb < K - 1:
                current_frontier = next_max
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='renorm'):
            ss = pl.slice(topk_w_t, [1, ROW_TILE], [0, r], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            for kb in pl.range(1, K):
                ss = pl.add(ss, pl.slice(topk_w_t, [1, ROW_TILE], [kb, r], valid_shape=[1, pl.min(ROW_TILE, M - r)]))
            selsum[0:1, r:r + ROW_TILE] = pl.slice(ss, [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)])
            inv = pl.div(pl.slice(pl.full([1, ROW_TILE], dtype=pl.FP32, value=SCALE), [1, ROW_TILE], [0, 0], valid_shape=[1, pl.min(ROW_TILE, M - r)]), pl.add(ss, 1e-20))
            wt_t = pl.slice(topk_w_t, [K, ROW_TILE], [0, r], valid_shape=[K, pl.min(ROW_TILE, M - r)])
            topk_w_t[0:K, r:r + ROW_TILE] = pl.slice(pl.col_expand_mul(wt_t, inv), [K, ROW_TILE], [0, 0], valid_shape=[K, pl.min(ROW_TILE, M - r)])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='owned_vector_weights'):
            export_weights = pl.slice(topk_w_t, [K, ROW_TILE], [0, r], valid_shape=[K, pl.min(ROW_TILE, M - r)])
            padded_weights = pl.full([16, ROW_TILE], dtype=pl.FP32, value=0.0)
            padded_weights = pl.assemble(padded_weights, export_weights, [0, 0])
            transposed_weights = pl.transpose(padded_weights, 0, 1)
            topk_w[r:r + ROW_TILE, 0:K] = pl.slice(transposed_weights, [ROW_TILE, K], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), K])
    for r in pl.parallel(0, M, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='owned_vector_indices'):
            export_indices = pl.slice(topk_idx_t, [K, ROW_TILE], [0, r], valid_shape=[K, pl.min(ROW_TILE, M - r)])
            padded_indices = pl.full([16, ROW_TILE], dtype=pl.INT32, value=0)
            padded_indices = pl.assemble(padded_indices, export_indices, [0, 0])
            transposed_indices = pl.transpose(padded_indices, 0, 1)
            topk_idx[r:r + ROW_TILE, 0:K] = pl.slice(transposed_indices, [ROW_TILE, K], [0, 0], valid_shape=[pl.min(ROW_TILE, M - r), K])
    return topk_w

def build(shape):
    global M, N, EPG, GROUP_WIDTH, RESIDENT_GROUP, INPUT_DTYPE, FRONTIER_ROWS, MASK_WIDTH, EXTRACT_WIDTH, EXTRACTION_BLOCKS
    import torch
    from golden import TensorSpec
    M, N = (int(shape['M']), int(shape['N']))
    if M <= 0 or N <= 0 or N % 128:
        raise ValueError('Expected positive M and positive N divisible by 128')
    requested_dtype = shape.get('dtype', 'fp32')
    if requested_dtype not in ('fp32', 'bf16'):
        raise ValueError('Unsupported dtype')
    INPUT_DTYPE = pl.BF16 if requested_dtype == 'bf16' else pl.FP32
    torch_input_dtype = torch.bfloat16 if requested_dtype == 'bf16' else torch.float32
    EPG = N // G
    width_limit = GROUP_ELEMENT_CAP // ROW_TILE
    GROUP_WIDTH = max((width for width in range(16, min(EPG, width_limit) + 1, 16) if EPG % width == 0))
    RESIDENT_GROUP = ROW_TILE * EPG <= GROUP_ELEMENT_CAP
    FRONTIER_ROWS = (M + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    producer_limit = min(EPG, MASK_WIDTH_CAP, PRODUCER_ELEMENT_CAP // ROW_TILE)
    MASK_WIDTH = max((width for width in (16, 32, 48, 64, 96, 128) if width <= producer_limit and EPG % width == 0))
    extraction_limit = min(N, EXTRACT_WIDTH_CAP, EXTRACTION_ELEMENT_CAP // EXTRACTION_ROWS)
    EXTRACT_WIDTH = max((width for width in range(128, extraction_limit + 1, 16) if N % width == 0))
    EXTRACTION_BLOCKS = (M + EXTRACTION_ROWS - 1) // EXTRACTION_ROWS
    rows, columns, experts_per_group = (M, N, EPG)

    def specs():
        return [TensorSpec('logits', [rows, columns], torch_input_dtype, init_value=torch.randn), TensorSpec('bias', [1, columns], torch.float32, init_value=torch.randn), TensorSpec('colidx', [1, columns], torch.float32, init_value=lambda: torch.arange(columns, dtype=torch.float32).reshape(1, columns)), TensorSpec('s', [rows, columns], torch.float32), TensorSpec('sel', [rows, columns], torch.float32), TensorSpec('smasked', [rows, columns], torch.float32), TensorSpec('gscore_t', [G, rows], torch.float32), TensorSpec('gmask_t', [G, rows], torch.float32), TensorSpec('topk_w_t', [K, rows], torch.float32), TensorSpec('topk_idx_t', [K, rows], torch.int32), TensorSpec('selsum', [1, rows], torch.float32), TensorSpec('topk_w', [rows, K], torch.float32), TensorSpec('topk_idx', [rows, K], torch.int32)]

    def golden(t):
        sig = torch.sigmoid(t['logits'].float())
        t['s'][:] = sig
        selection = sig + t['bias']
        t['sel'][:] = selection
        grouped = selection.view(rows, G, experts_per_group)
        first = grouped.max(dim=-1, keepdim=True).values
        suppressed = grouped - NEG * (grouped == first).float()
        scores = first.squeeze(-1) + suppressed.max(dim=-1).values
        group_mask = torch.zeros(rows, G)
        for _ in range(TG):
            maximum = scores.max(dim=-1, keepdim=True).values
            winners = (scores == maximum).float()
            group_mask = group_mask + winners
            scores = scores - NEG * winners
        t['gmask_t'][:] = group_mask.t()
        t['gscore_t'][:] = scores.t()
        expanded_mask = group_mask.repeat_interleave(experts_per_group, dim=1)
        masked = selection * expanded_mask - NEG * (1.0 - expanded_mask)
        weights = torch.zeros(rows, K)
        indices = torch.zeros(rows, K, dtype=torch.int32)
        columns_index = torch.arange(columns, dtype=torch.float32)
        for rank in range(K):
            maximum = masked.max(dim=-1, keepdim=True).values
            winners = (masked == maximum).float()
            indices[:, rank] = (winners * columns_index).sum(-1).to(torch.int32)
            weights[:, rank] = (winners * sig).sum(-1)
            masked = masked - NEG * winners
        t['smasked'][:] = masked
        t['topk_idx_t'][:] = indices.t()
        weight_sum = weights.sum(dim=-1, keepdim=True)
        t['selsum'][:] = weight_sum.reshape(1, -1)
        weights = weights * (SCALE / (weight_sum + 1e-20))
        t['topk_w_t'][:] = weights.t()
        t['topk_w'][:] = weights
        t['topk_idx'][:] = indices
    return (grouped_topk_bf16 if requested_dtype == 'bf16' else grouped_topk_fp32, specs(), golden, RTOL, ATOL)

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
