# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact resident Gumbel sampling with bounded workspace residency."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


TAU = 0.5
K = 16
NEG = -3.0e38
MASK_STEP = 1.0e30
EXTRACT_ROW_TILE = 8
COL_TILE = 128
RESIDENT_ELEMENT_CAP = 8192
EXTRACTION_UNROLL = 4
ROW_GROUPS = 1


def _aligned_divisor(extent, requested, alignment):
    tile = min(extent, requested)
    tile -= tile % alignment
    while extent % tile:
        tile -= alignment
    return tile


def build(shape):
    import torch
    from golden import TensorSpec

    rows, cols = int(shape["M"]), int(shape["N"])
    if rows <= 0 or cols <= 0 or rows % 64 or cols % 128:
        raise ValueError("M must be a positive multiple of 64 and N of 128")
    requested_dtype = shape.get("dtype", "fp32")
    if requested_dtype not in ("fp32", "bf16"):
        raise ValueError("dtype must be fp32 or bf16")
    logits_dtype = pl.BF16 if requested_dtype == "bf16" else pl.FP32
    torch_logits_dtype = torch.bfloat16 if requested_dtype == "bf16" else torch.float32
    row_tile = _aligned_divisor(rows, EXTRACT_ROW_TILE, 8)
    while row_tile > 8 and row_tile * cols > RESIDENT_ELEMENT_CAP:
        row_tile = _aligned_divisor(rows, row_tile - 8, 8)
    col_tile = _aligned_divisor(cols, COL_TILE, 128)
    task_row_tile = _aligned_divisor(rows, row_tile * ROW_GROUPS, row_tile)
    resident = row_tile * cols <= RESIDENT_ELEMENT_CAP
    index_sentinel = float(cols)

    if requested_dtype == "bf16":
        if resident:
            @pl.jit
            def sample_resident(
                logits: pl.Tensor[[rows, cols], logits_dtype],
                noise: pl.Tensor[[rows, cols], pl.FP32],
                colidx: pl.Tensor[[1, cols], pl.FP32],
                scaled: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                p: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                thresh: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                filtered: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                g: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                rowmax: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                token_ids: pl.Out[pl.Tensor[[rows, 1], pl.INT32]],
            ):
                for group_start in pl.parallel(0, rows, task_row_tile):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_sample"):
                        for row_offset in pl.range(0, task_row_tile, row_tile):
                            row_start = group_start + row_offset
                            scores = pl.div(pl.cast(logits[row_start:row_start + row_tile, :], target_type=pl.FP32), TAU)
                            remaining = pl.mul(scores, 1.0)
                            threshold = pl.row_max(remaining)
                            for extraction_group in pl.range(K // EXTRACTION_UNROLL):
                                for extraction_step in pl.unroll(EXTRACTION_UNROLL):
                                    threshold = pl.row_max(remaining)
                                    threshold_full = pl.row_expand(remaining, threshold)
                                    equal = pl.cmp(remaining, threshold_full, cmp_type=0)
                                    remaining = pl.sub(remaining, pl.mul(equal, MASK_STEP))
                            scaled[row_start:row_start + row_tile, :] = scores
                            p[row_start:row_start + row_tile, :] = remaining
                            thresh[0:1, row_start:row_start + row_tile] = pl.reshape(threshold, [1, row_tile])
                            threshold_full = pl.row_expand(scores, threshold)
                            keep = pl.cmp(scores, threshold_full, cmp_type=5)
                            drop = pl.cmp(scores, threshold_full, cmp_type=2)
                            filtered_scores = pl.add(pl.mul(scores, keep), pl.mul(drop, NEG))
                            filtered[row_start:row_start + row_tile, :] = filtered_scores
                            perturbed = pl.add(filtered_scores, noise[row_start:row_start + row_tile, :])
                            g[row_start:row_start + row_tile, :] = perturbed
                            maxima = pl.row_max(perturbed)
                            rowmax[0:1, row_start:row_start + row_tile] = pl.reshape(maxima, [1, row_tile])
                            maxima_full = pl.row_expand(perturbed, maxima)
                            equal = pl.cmp(perturbed, maxima_full, cmp_type=0)
                            ones = pl.full([row_tile, cols], dtype=pl.FP32, value=1.0)
                            indices = pl.col_expand_mul(ones, colidx[0:1, :])
                            candidates = pl.add(pl.mul(equal, pl.sub(indices, index_sentinel)), index_sentinel)
                            winner = pl.row_min(candidates)
                            token_ids[row_start:row_start + row_tile, 0:1] = pl.cast(winner, target_type=pl.INT32, mode="rint")
                return token_ids

            fn = sample_resident

        else:
            @pl.jit
            def sample_streamed(
                logits: pl.Tensor[[rows, cols], logits_dtype],
                noise: pl.Tensor[[rows, cols], pl.FP32],
                colidx: pl.Tensor[[1, cols], pl.FP32],
                scaled: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                p: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                thresh: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                filtered: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                g: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                rowmax: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                token_ids: pl.Out[pl.Tensor[[rows, 1], pl.INT32]],
            ):
                for row_start in pl.parallel(0, rows, row_tile):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stream_scale"):
                        for col_block in pl.range(cols // col_tile):
                            col_start = col_block * col_tile
                            scores = pl.div(pl.cast(logits[row_start:row_start + row_tile, col_start:col_start + col_tile], target_type=pl.FP32), TAU)
                            scaled[row_start:row_start + row_tile, col_start:col_start + col_tile] = scores
                            p[row_start:row_start + row_tile, col_start:col_start + col_tile] = scores
                for extraction in pl.range(K):
                    for row_start in pl.parallel(0, rows, row_tile):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="stream_extract_max"):
                            extraction_maxima = pl.full([1, row_tile], dtype=pl.FP32, value=-3.4028234663852886e38)
                            for col_block in pl.range(cols // col_tile):
                                col_start = col_block * col_tile
                                chunk = p[row_start:row_start + row_tile, col_start:col_start + col_tile]
                                extraction_maxima = pl.maximum(extraction_maxima, pl.reshape(pl.row_max(chunk), [1, row_tile]))
                            thresh[0:1, row_start:row_start + row_tile] = extraction_maxima
                    for row_start in pl.parallel(0, rows, row_tile):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="stream_extract_mask"):
                            threshold = pl.reshape(thresh[0:1, row_start:row_start + row_tile], [row_tile, 1])
                            zeros = pl.full([row_tile, col_tile], dtype=pl.FP32, value=0.0)
                            threshold_full = pl.row_expand(zeros, threshold)
                            for col_block in pl.range(cols // col_tile):
                                col_start = col_block * col_tile
                                chunk = p[row_start:row_start + row_tile, col_start:col_start + col_tile]
                                equal = pl.cmp(chunk, threshold_full, cmp_type=0)
                                p[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.sub(chunk, pl.mul(equal, MASK_STEP))
                for row_start in pl.parallel(0, rows, row_tile):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="chunked_sample"):
                        best_scores = pl.full([1, row_tile], dtype=pl.FP32, value=-3.4028234663852886e38)
                        best_indices = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                        threshold = pl.reshape(thresh[0:1, row_start:row_start + row_tile], [row_tile, 1])
                        chunk_zeros = pl.full([row_tile, col_tile], dtype=pl.FP32, value=0.0)
                        ones = pl.full([row_tile, col_tile], dtype=pl.FP32, value=1.0)
                        chunk_threshold_full = pl.row_expand(chunk_zeros, threshold)
                        for col_block in pl.range(cols // col_tile):
                            col_start = col_block * col_tile
                            chunk_scores = scaled[row_start:row_start + row_tile, col_start:col_start + col_tile]
                            keep = pl.cmp(chunk_scores, chunk_threshold_full, cmp_type=5)
                            drop = pl.cmp(chunk_scores, chunk_threshold_full, cmp_type=2)
                            filtered_scores = pl.add(pl.mul(chunk_scores, keep), pl.mul(drop, NEG))
                            filtered[row_start:row_start + row_tile, col_start:col_start + col_tile] = filtered_scores
                            perturbed = pl.add(filtered_scores, noise[row_start:row_start + row_tile, col_start:col_start + col_tile])
                            g[row_start:row_start + row_tile, col_start:col_start + col_tile] = perturbed
                            maxima = pl.row_max(perturbed)
                            maxima_full = pl.row_expand(chunk_zeros, maxima)
                            chunk_equal = pl.cmp(perturbed, maxima_full, cmp_type=0)
                            indices = pl.col_expand_mul(ones, colidx[0:1, col_start:col_start + col_tile])
                            candidates = pl.add(pl.mul(chunk_equal, pl.sub(indices, index_sentinel)), index_sentinel)
                            local_winner = pl.reshape(pl.row_min(candidates), [1, row_tile])
                            local_scores = pl.reshape(maxima, [1, row_tile])
                            newer = pl.cmp(local_scores, best_scores, cmp_type=4)
                            best_indices = pl.add(best_indices, pl.mul(newer, pl.sub(local_winner, best_indices)))
                            best_scores = pl.maximum(best_scores, local_scores)
                        rowmax[0:1, row_start:row_start + row_tile] = best_scores
                        token_ids[row_start:row_start + row_tile, 0:1] = pl.reshape(pl.cast(best_indices, target_type=pl.INT32, mode="rint"), [row_tile, 1])
                return token_ids

            fn = sample_streamed
    else:
        if resident:
            @pl.jit
            def sample_resident(
                logits: pl.Tensor[[rows, cols], logits_dtype],
                noise: pl.Tensor[[rows, cols], pl.FP32],
                colidx: pl.Tensor[[1, cols], pl.FP32],
                scaled: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                p: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                thresh: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                filtered: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                g: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                rowmax: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                token_ids: pl.Out[pl.Tensor[[rows, 1], pl.INT32]],
            ):
                for group_start in pl.parallel(0, rows, task_row_tile):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="resident_sample"):
                        for row_offset in pl.range(0, task_row_tile, row_tile):
                            row_start = group_start + row_offset
                            scores = pl.div(logits[row_start:row_start + row_tile, :], TAU)
                            remaining = pl.mul(scores, 1.0)
                            threshold = pl.row_max(remaining)
                            for extraction_group in pl.range(K // EXTRACTION_UNROLL):
                                for extraction_step in pl.unroll(EXTRACTION_UNROLL):
                                    threshold = pl.row_max(remaining)
                                    threshold_full = pl.row_expand(remaining, threshold)
                                    equal = pl.cmp(remaining, threshold_full, cmp_type=0)
                                    remaining = pl.sub(remaining, pl.mul(equal, MASK_STEP))
                            scaled[row_start:row_start + row_tile, :] = scores
                            p[row_start:row_start + row_tile, :] = remaining
                            thresh[0:1, row_start:row_start + row_tile] = pl.reshape(threshold, [1, row_tile])
                            threshold_full = pl.row_expand(scores, threshold)
                            keep = pl.cmp(scores, threshold_full, cmp_type=5)
                            drop = pl.cmp(scores, threshold_full, cmp_type=2)
                            filtered_scores = pl.add(pl.mul(scores, keep), pl.mul(drop, NEG))
                            filtered[row_start:row_start + row_tile, :] = filtered_scores
                            perturbed = pl.add(filtered_scores, noise[row_start:row_start + row_tile, :])
                            g[row_start:row_start + row_tile, :] = perturbed
                            maxima = pl.row_max(perturbed)
                            rowmax[0:1, row_start:row_start + row_tile] = pl.reshape(maxima, [1, row_tile])
                            maxima_full = pl.row_expand(perturbed, maxima)
                            equal = pl.cmp(perturbed, maxima_full, cmp_type=0)
                            ones = pl.full([row_tile, cols], dtype=pl.FP32, value=1.0)
                            indices = pl.col_expand_mul(ones, colidx[0:1, :])
                            candidates = pl.add(pl.mul(equal, pl.sub(indices, index_sentinel)), index_sentinel)
                            winner = pl.row_min(candidates)
                            token_ids[row_start:row_start + row_tile, 0:1] = pl.cast(winner, target_type=pl.INT32, mode="rint")
                return token_ids

            fn = sample_resident

        else:
            @pl.jit
            def sample_streamed(
                logits: pl.Tensor[[rows, cols], logits_dtype],
                noise: pl.Tensor[[rows, cols], pl.FP32],
                colidx: pl.Tensor[[1, cols], pl.FP32],
                scaled: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                p: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                thresh: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                filtered: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                g: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
                rowmax: pl.InOut[pl.Tensor[[1, rows], pl.FP32]],
                token_ids: pl.Out[pl.Tensor[[rows, 1], pl.INT32]],
            ):
                for row_start in pl.parallel(0, rows, row_tile):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stream_scale"):
                        for col_block in pl.range(cols // col_tile):
                            col_start = col_block * col_tile
                            scores = pl.div(logits[row_start:row_start + row_tile, col_start:col_start + col_tile], TAU)
                            scaled[row_start:row_start + row_tile, col_start:col_start + col_tile] = scores
                            p[row_start:row_start + row_tile, col_start:col_start + col_tile] = scores
                for extraction in pl.range(K):
                    for row_start in pl.parallel(0, rows, row_tile):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="stream_extract_max"):
                            extraction_maxima = pl.full([1, row_tile], dtype=pl.FP32, value=-3.4028234663852886e38)
                            for col_block in pl.range(cols // col_tile):
                                col_start = col_block * col_tile
                                chunk = p[row_start:row_start + row_tile, col_start:col_start + col_tile]
                                extraction_maxima = pl.maximum(extraction_maxima, pl.reshape(pl.row_max(chunk), [1, row_tile]))
                            thresh[0:1, row_start:row_start + row_tile] = extraction_maxima
                    for row_start in pl.parallel(0, rows, row_tile):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="stream_extract_mask"):
                            threshold = pl.reshape(thresh[0:1, row_start:row_start + row_tile], [row_tile, 1])
                            zeros = pl.full([row_tile, col_tile], dtype=pl.FP32, value=0.0)
                            threshold_full = pl.row_expand(zeros, threshold)
                            for col_block in pl.range(cols // col_tile):
                                col_start = col_block * col_tile
                                chunk = p[row_start:row_start + row_tile, col_start:col_start + col_tile]
                                equal = pl.cmp(chunk, threshold_full, cmp_type=0)
                                p[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.sub(chunk, pl.mul(equal, MASK_STEP))
                for row_start in pl.parallel(0, rows, row_tile):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="chunked_sample"):
                        best_scores = pl.full([1, row_tile], dtype=pl.FP32, value=-3.4028234663852886e38)
                        best_indices = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                        threshold = pl.reshape(thresh[0:1, row_start:row_start + row_tile], [row_tile, 1])
                        chunk_zeros = pl.full([row_tile, col_tile], dtype=pl.FP32, value=0.0)
                        ones = pl.full([row_tile, col_tile], dtype=pl.FP32, value=1.0)
                        chunk_threshold_full = pl.row_expand(chunk_zeros, threshold)
                        for col_block in pl.range(cols // col_tile):
                            col_start = col_block * col_tile
                            chunk_scores = scaled[row_start:row_start + row_tile, col_start:col_start + col_tile]
                            keep = pl.cmp(chunk_scores, chunk_threshold_full, cmp_type=5)
                            drop = pl.cmp(chunk_scores, chunk_threshold_full, cmp_type=2)
                            filtered_scores = pl.add(pl.mul(chunk_scores, keep), pl.mul(drop, NEG))
                            filtered[row_start:row_start + row_tile, col_start:col_start + col_tile] = filtered_scores
                            perturbed = pl.add(filtered_scores, noise[row_start:row_start + row_tile, col_start:col_start + col_tile])
                            g[row_start:row_start + row_tile, col_start:col_start + col_tile] = perturbed
                            maxima = pl.row_max(perturbed)
                            maxima_full = pl.row_expand(chunk_zeros, maxima)
                            chunk_equal = pl.cmp(perturbed, maxima_full, cmp_type=0)
                            indices = pl.col_expand_mul(ones, colidx[0:1, col_start:col_start + col_tile])
                            candidates = pl.add(pl.mul(chunk_equal, pl.sub(indices, index_sentinel)), index_sentinel)
                            local_winner = pl.reshape(pl.row_min(candidates), [1, row_tile])
                            local_scores = pl.reshape(maxima, [1, row_tile])
                            newer = pl.cmp(local_scores, best_scores, cmp_type=4)
                            best_indices = pl.add(best_indices, pl.mul(newer, pl.sub(local_winner, best_indices)))
                            best_scores = pl.maximum(best_scores, local_scores)
                        rowmax[0:1, row_start:row_start + row_tile] = best_scores
                        token_ids[row_start:row_start + row_tile, 0:1] = pl.reshape(pl.cast(best_indices, target_type=pl.INT32, mode="rint"), [row_tile, 1])
                return token_ids

            fn = sample_streamed

    def noise_input():
        generator = torch.Generator().manual_seed(20261106)
        uniform = torch.rand(rows, cols, generator=generator, dtype=torch.float32).clamp_min(1.0e-6)
        return -torch.log(-torch.log(uniform))

    specs = [
        TensorSpec("logits", [rows, cols], torch_logits_dtype, init_value=torch.randn),
        TensorSpec("noise", [rows, cols], torch.float32, init_value=noise_input),
        TensorSpec("colidx", [1, cols], torch.float32, init_value=lambda: torch.arange(cols, dtype=torch.float32).reshape(1, cols)),
        TensorSpec("scaled", [rows, cols], torch.float32),
        TensorSpec("p", [rows, cols], torch.float32),
        TensorSpec("thresh", [1, rows], torch.float32),
        TensorSpec("filtered", [rows, cols], torch.float32),
        TensorSpec("g", [rows, cols], torch.float32),
        TensorSpec("rowmax", [1, rows], torch.float32),
        TensorSpec("token_ids", [rows, 1], torch.int32),
    ]

    def golden(tensors):
        scores = tensors["logits"].float() / TAU
        tensors["scaled"][:] = scores
        remaining = scores.clone()
        for extraction in range(K):
            threshold = remaining.max(dim=-1, keepdim=True).values
            remaining = remaining - MASK_STEP * (remaining == threshold).to(remaining.dtype)
        tensors["p"][:] = remaining
        tensors["thresh"][:] = threshold.reshape(1, rows)
        filtered_scores = torch.where(scores >= threshold, scores, torch.full_like(scores, NEG))
        tensors["filtered"][:] = filtered_scores
        perturbed = filtered_scores + tensors["noise"]
        tensors["g"][:] = perturbed
        tensors["rowmax"][:] = perturbed.max(dim=-1, keepdim=True).values.reshape(1, rows)
        tensors["token_ids"][:] = torch.argmax(perturbed, dim=-1).reshape(rows, 1).to(torch.int32)

    return fn, specs, golden, 0.0, 0.0

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
