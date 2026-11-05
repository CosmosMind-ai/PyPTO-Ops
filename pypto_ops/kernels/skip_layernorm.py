# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused Add + LayerNorm (skip_layernorm) — outputs h, xhat, mu, rstd, y; gamma/beta [1, N]; eps=1e-5; bf16 storage with fp32 arithmetic, fp32 path kept. build() shape-dispatches among the resident/streamed variants; all are kept."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}

ROW_TILE = 32
STATS_COL_TILE = 128
APPLY_COL_TILE = 256
STATS_MODE = 1
DEFERRED = 1
STATS_DTYPE = 'fp32'
COMPUTE_DTYPE = 'fp32'
FP32_STATS_MODE = 0
EPS = 1e-05

@pl.jit
def skip_layernorm_bf16(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='skip_layernorm_rows'):
            valid_rows = pl.min(ROW_TILE, rows - row_start)
            sum_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            sq_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            for scan0_col_start in pl.range(0, hidden, STATS_COL_TILE):
                scan0_valid_cols = pl.min(STATS_COL_TILE, hidden - scan0_col_start)
                scan0_x_tile = pl.slice(x, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_res_tile = pl.slice(res, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_residual = pl.add(pl.cast(scan0_x_tile, pl.FP32), pl.cast(scan0_res_tile, pl.FP32))
                scan0_residual = pl.fillpad(pl.set_validshape(scan0_residual, valid_rows, scan0_valid_cols), pad_value=pl.PadValue.zero)
                scan0_residual = pl.set_validshape(scan0_residual, ROW_TILE, STATS_COL_TILE)
                sum_acc = pl.add(sum_acc, scan0_residual)
                sq_acc = pl.add(sq_acc, pl.mul(scan0_residual, scan0_residual))
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, ROW_TILE])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [ROW_TILE, 1])
            square_row = pl.reshape(pl.row_sum(sq_acc), [1, ROW_TILE])
            variance_row = pl.sub(pl.mul(square_row, 1.0 / hidden), pl.mul(mean_row, mean_row))
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [ROW_TILE, 1])
            mu[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(inv_std_row, 1, valid_rows)
            for scan1_col_start in pl.range(0, hidden, APPLY_COL_TILE):
                scan1_valid_cols = pl.min(APPLY_COL_TILE, hidden - scan1_col_start)
                scan1_x_tile = pl.slice(x, [ROW_TILE, APPLY_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_res_tile = pl.slice(res, [ROW_TILE, APPLY_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_gamma_tile = pl.slice(gamma, [1, APPLY_COL_TILE], [0, scan1_col_start], valid_shape=[1, scan1_valid_cols])
                scan1_beta_tile = pl.slice(beta, [1, APPLY_COL_TILE], [0, scan1_col_start], valid_shape=[1, scan1_valid_cols])
                scan1_residual = pl.add(pl.cast(scan1_x_tile, pl.FP32), pl.cast(scan1_res_tile, pl.FP32))
                scan1_normalized = pl.row_expand_mul(pl.row_expand_sub(scan1_residual, mean_col), inv_std_col)
                scan1_result = pl.col_expand_add(pl.col_expand_mul(scan1_normalized, pl.cast(scan1_gamma_tile, pl.FP32)), pl.cast(scan1_beta_tile, pl.FP32))
                h[row_start:row_start + ROW_TILE, scan1_col_start:scan1_col_start + APPLY_COL_TILE] = pl.set_validshape(pl.cast(scan1_residual, pl.BF16), valid_rows, scan1_valid_cols)
                xhat[row_start:row_start + ROW_TILE, scan1_col_start:scan1_col_start + APPLY_COL_TILE] = pl.set_validshape(pl.cast(scan1_normalized, pl.BF16), valid_rows, scan1_valid_cols)
                y[row_start:row_start + ROW_TILE, scan1_col_start:scan1_col_start + APPLY_COL_TILE] = pl.set_validshape(pl.cast(scan1_result, pl.BF16), valid_rows, scan1_valid_cols)
    return y

@pl.jit
def skip_layernorm_bf16_aligned(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='skip_layernorm_rows'):
            valid_rows = ROW_TILE
            sum_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            sq_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            for scan0_col_start in pl.range(0, hidden, STATS_COL_TILE):
                scan0_valid_cols = STATS_COL_TILE
                scan0_x_tile = pl.slice(x, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_res_tile = pl.slice(res, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_residual = pl.add(pl.cast(scan0_x_tile, pl.FP32), pl.cast(scan0_res_tile, pl.FP32))
                sum_acc = pl.add(sum_acc, scan0_residual)
                sq_acc = pl.add(sq_acc, pl.mul(scan0_residual, scan0_residual))
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, ROW_TILE])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [ROW_TILE, 1])
            square_row = pl.reshape(pl.row_sum(sq_acc), [1, ROW_TILE])
            variance_row = pl.sub(pl.mul(square_row, 1.0 / hidden), pl.mul(mean_row, mean_row))
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [ROW_TILE, 1])
            mu[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(inv_std_row, 1, valid_rows)
            for scan1_col_start in pl.range(0, hidden, APPLY_COL_TILE):
                scan1_valid_cols = APPLY_COL_TILE
                scan1_x_tile = pl.slice(x, [ROW_TILE, APPLY_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_res_tile = pl.slice(res, [ROW_TILE, APPLY_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_gamma_tile = pl.slice(gamma, [1, APPLY_COL_TILE], [0, scan1_col_start], valid_shape=[1, scan1_valid_cols])
                scan1_beta_tile = pl.slice(beta, [1, APPLY_COL_TILE], [0, scan1_col_start], valid_shape=[1, scan1_valid_cols])
                scan1_residual = pl.add(pl.cast(scan1_x_tile, pl.FP32), pl.cast(scan1_res_tile, pl.FP32))
                scan1_normalized = pl.row_expand_mul(pl.row_expand_sub(scan1_residual, mean_col), inv_std_col)
                scan1_result = pl.col_expand_add(pl.col_expand_mul(scan1_normalized, pl.cast(scan1_gamma_tile, pl.FP32)), pl.cast(scan1_beta_tile, pl.FP32))
                h[row_start:row_start + ROW_TILE, scan1_col_start:scan1_col_start + APPLY_COL_TILE] = pl.set_validshape(pl.cast(scan1_residual, pl.BF16), valid_rows, scan1_valid_cols)
                xhat[row_start:row_start + ROW_TILE, scan1_col_start:scan1_col_start + APPLY_COL_TILE] = pl.set_validshape(pl.cast(scan1_normalized, pl.BF16), valid_rows, scan1_valid_cols)
                y[row_start:row_start + ROW_TILE, scan1_col_start:scan1_col_start + APPLY_COL_TILE] = pl.set_validshape(pl.cast(scan1_result, pl.BF16), valid_rows, scan1_valid_cols)
    return y

@pl.jit
def skip_layernorm_fp32(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='skip_layernorm_rows'):
            valid_rows = pl.min(ROW_TILE, rows - row_start)
            sum_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            sq_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            for scan0_col_start in pl.range(0, hidden, STATS_COL_TILE):
                scan0_valid_cols = pl.min(STATS_COL_TILE, hidden - scan0_col_start)
                scan0_x_tile = pl.slice(x, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_res_tile = pl.slice(res, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_residual = pl.add(scan0_x_tile, scan0_res_tile)
                scan0_residual = pl.fillpad(pl.set_validshape(scan0_residual, valid_rows, scan0_valid_cols), pad_value=pl.PadValue.zero)
                scan0_residual = pl.set_validshape(scan0_residual, ROW_TILE, STATS_COL_TILE)
                sum_acc = pl.add(sum_acc, scan0_residual)
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, ROW_TILE])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [ROW_TILE, 1])
            centred_sq_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            for scan1_col_start in pl.range(0, hidden, STATS_COL_TILE):
                scan1_valid_cols = pl.min(STATS_COL_TILE, hidden - scan1_col_start)
                scan1_x_tile = pl.slice(x, [ROW_TILE, STATS_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_res_tile = pl.slice(res, [ROW_TILE, STATS_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_residual = pl.add(scan1_x_tile, scan1_res_tile)
                scan1_centred = pl.row_expand_sub(scan1_residual, mean_col)
                scan1_centred = pl.fillpad(pl.set_validshape(scan1_centred, valid_rows, scan1_valid_cols), pad_value=pl.PadValue.zero)
                scan1_centred = pl.set_validshape(scan1_centred, ROW_TILE, STATS_COL_TILE)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan1_centred, scan1_centred))
            centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, ROW_TILE])
            variance_row = pl.mul(centred_square_row, 1.0 / hidden)
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [ROW_TILE, 1])
            mu[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(inv_std_row, 1, valid_rows)
            for scan2_col_start in pl.range(0, hidden, APPLY_COL_TILE):
                scan2_valid_cols = pl.min(APPLY_COL_TILE, hidden - scan2_col_start)
                scan2_x_tile = pl.slice(x, [ROW_TILE, APPLY_COL_TILE], [row_start, scan2_col_start], valid_shape=[valid_rows, scan2_valid_cols])
                scan2_res_tile = pl.slice(res, [ROW_TILE, APPLY_COL_TILE], [row_start, scan2_col_start], valid_shape=[valid_rows, scan2_valid_cols])
                scan2_gamma_tile = pl.slice(gamma, [1, APPLY_COL_TILE], [0, scan2_col_start], valid_shape=[1, scan2_valid_cols])
                scan2_beta_tile = pl.slice(beta, [1, APPLY_COL_TILE], [0, scan2_col_start], valid_shape=[1, scan2_valid_cols])
                scan2_residual = pl.add(scan2_x_tile, scan2_res_tile)
                scan2_normalized = pl.row_expand_mul(pl.row_expand_sub(scan2_residual, mean_col), inv_std_col)
                scan2_result = pl.col_expand_add(pl.col_expand_mul(scan2_normalized, scan2_gamma_tile), scan2_beta_tile)
                h[row_start:row_start + ROW_TILE, scan2_col_start:scan2_col_start + APPLY_COL_TILE] = pl.set_validshape(scan2_residual, valid_rows, scan2_valid_cols)
                xhat[row_start:row_start + ROW_TILE, scan2_col_start:scan2_col_start + APPLY_COL_TILE] = pl.set_validshape(scan2_normalized, valid_rows, scan2_valid_cols)
                y[row_start:row_start + ROW_TILE, scan2_col_start:scan2_col_start + APPLY_COL_TILE] = pl.set_validshape(scan2_result, valid_rows, scan2_valid_cols)
    return y

@pl.jit
def skip_layernorm_fp32_aligned(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='skip_layernorm_rows'):
            valid_rows = ROW_TILE
            sum_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            sq_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            for scan0_col_start in pl.range(0, hidden, STATS_COL_TILE):
                scan0_valid_cols = STATS_COL_TILE
                scan0_x_tile = pl.slice(x, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_res_tile = pl.slice(res, [ROW_TILE, STATS_COL_TILE], [row_start, scan0_col_start], valid_shape=[valid_rows, scan0_valid_cols])
                scan0_residual = pl.add(scan0_x_tile, scan0_res_tile)
                sum_acc = pl.add(sum_acc, scan0_residual)
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, ROW_TILE])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [ROW_TILE, 1])
            centred_sq_acc = pl.full([ROW_TILE, STATS_COL_TILE], dtype=pl.FP32, value=0.0)
            for scan1_col_start in pl.range(0, hidden, STATS_COL_TILE):
                scan1_valid_cols = STATS_COL_TILE
                scan1_x_tile = pl.slice(x, [ROW_TILE, STATS_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_res_tile = pl.slice(res, [ROW_TILE, STATS_COL_TILE], [row_start, scan1_col_start], valid_shape=[valid_rows, scan1_valid_cols])
                scan1_residual = pl.add(scan1_x_tile, scan1_res_tile)
                scan1_centred = pl.row_expand_sub(scan1_residual, mean_col)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan1_centred, scan1_centred))
            centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, ROW_TILE])
            variance_row = pl.mul(centred_square_row, 1.0 / hidden)
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [ROW_TILE, 1])
            mu[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(inv_std_row, 1, valid_rows)
            for scan2_col_start in pl.range(0, hidden, APPLY_COL_TILE):
                scan2_valid_cols = APPLY_COL_TILE
                scan2_x_tile = pl.slice(x, [ROW_TILE, APPLY_COL_TILE], [row_start, scan2_col_start], valid_shape=[valid_rows, scan2_valid_cols])
                scan2_res_tile = pl.slice(res, [ROW_TILE, APPLY_COL_TILE], [row_start, scan2_col_start], valid_shape=[valid_rows, scan2_valid_cols])
                scan2_gamma_tile = pl.slice(gamma, [1, APPLY_COL_TILE], [0, scan2_col_start], valid_shape=[1, scan2_valid_cols])
                scan2_beta_tile = pl.slice(beta, [1, APPLY_COL_TILE], [0, scan2_col_start], valid_shape=[1, scan2_valid_cols])
                scan2_residual = pl.add(scan2_x_tile, scan2_res_tile)
                scan2_normalized = pl.row_expand_mul(pl.row_expand_sub(scan2_residual, mean_col), inv_std_col)
                scan2_result = pl.col_expand_add(pl.col_expand_mul(scan2_normalized, scan2_gamma_tile), scan2_beta_tile)
                h[row_start:row_start + ROW_TILE, scan2_col_start:scan2_col_start + APPLY_COL_TILE] = pl.set_validshape(scan2_residual, valid_rows, scan2_valid_cols)
                xhat[row_start:row_start + ROW_TILE, scan2_col_start:scan2_col_start + APPLY_COL_TILE] = pl.set_validshape(scan2_normalized, valid_rows, scan2_valid_cols)
                y[row_start:row_start + ROW_TILE, scan2_col_start:scan2_col_start + APPLY_COL_TILE] = pl.set_validshape(scan2_result, valid_rows, scan2_valid_cols)
    return y

CACHE_LIMIT = 4096
CACHE_QUANTUM = 256
CACHE_MODE = 0
MIN_CACHED_FRACTION_Q8 = 0


@pl.jit
def singleton_bf16(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="singleton_rows"):
            valid_rows = pl.min(ROW_TILE, rows - row_start)
            x_tile = pl.slice(x, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            res_tile = pl.slice(res, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            x_tile = pl.set_validshape(x_tile, ROW_TILE, 1)
            res_tile = pl.set_validshape(res_tile, ROW_TILE, 1)
            residual = pl.add(pl.cast(x_tile, pl.FP32), pl.cast(res_tile, pl.FP32))
            mean_row = pl.reshape(residual, [1, ROW_TILE])
            inv_std_row = pl.rsqrt(pl.full([1, ROW_TILE], dtype=pl.FP32, value=EPS), high_precision=True)
            normalized_row = pl.mul(pl.sub(mean_row, mean_row), 316.2277660168379)
            gamma_tile = pl.reshape(pl.slice(gamma, [1, 1], [0, 0]), [1, 1])
            beta_tile = pl.reshape(pl.slice(beta, [1, 1], [0, 0]), [1, 1])
            gamma_scalar = pl.read(gamma_tile, [0, 0])
            beta_scalar = pl.read(beta_tile, [0, 0])
            affine_row = pl.add(pl.mul(normalized_row, pl.cast(gamma_scalar, pl.FP32)), pl.cast(beta_scalar, pl.FP32))
            normalized = pl.reshape(normalized_row, [ROW_TILE, 1])
            affine = pl.reshape(affine_row, [ROW_TILE, 1])
            h[row_start:row_start + ROW_TILE, 0:1] = pl.set_validshape(pl.cast(residual, pl.BF16), valid_rows, 1)
            xhat[row_start:row_start + ROW_TILE, 0:1] = pl.set_validshape(pl.cast(normalized, pl.BF16), valid_rows, 1)
            y[row_start:row_start + ROW_TILE, 0:1] = pl.set_validshape(pl.cast(affine, pl.BF16), valid_rows, 1)
            mu[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(inv_std_row, 1, valid_rows)
    return y


@pl.jit
def peeled_bf16(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows // 32 * 32, 32):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='interior_rows'):
            sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan0_col in pl.range(0, hidden // 128 * 128, 128):
                scan0_x = pl.load(x, [row_start, scan0_col], [32, 128], valid_shape=[32, 128])
                scan0_res = pl.load(res, [row_start, scan0_col], [32, 128], valid_shape=[32, 128])
                scan0_u = pl.add(pl.cast(scan0_x, pl.FP32), pl.cast(scan0_res, pl.FP32))
                sum_acc = pl.add(sum_acc, scan0_u)
                sq_acc = pl.add(sq_acc, pl.mul(scan0_u, scan0_u))
            if hidden % 128 != 0 and 0 < hidden:
                for scan1_col in pl.range(hidden // 128 * 128, hidden, 128):
                    scan1_cols = pl.min(128, hidden - scan1_col)
                    scan1_x = pl.load(x, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                    scan1_res = pl.load(res, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                    scan1_u = pl.add(pl.cast(scan1_x, pl.FP32), pl.cast(scan1_res, pl.FP32))
                    scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                    scan1_u = pl.set_validshape(scan1_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan1_u)
                    sq_acc = pl.add(sq_acc, pl.mul(scan1_u, scan1_u))
            square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_square_row = pl.mul(square_row, 1.0 / hidden)
            sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
            inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            pl.store(pl.set_validshape(mean_row, 1, 32), [0, row_start], mu)
            pl.store(pl.set_validshape(inv_std_row, 1, 32), [0, row_start], rstd)
            for scan2_col in pl.range(0, hidden // 256 * 256, 256):
                scan2_x = pl.load(x, [row_start, scan2_col], [32, 256], valid_shape=[32, 256])
                scan2_res = pl.load(res, [row_start, scan2_col], [32, 256], valid_shape=[32, 256])
                scan2_u = pl.add(pl.cast(scan2_x, pl.FP32), pl.cast(scan2_res, pl.FP32))
                scan2_gamma = pl.load(gamma, [0, scan2_col], [1, 256], valid_shape=[1, 256])
                scan2_beta = pl.load(beta, [0, scan2_col], [1, 256], valid_shape=[1, 256])
                scan2_normalized = pl.row_expand_mul(pl.row_expand_sub(scan2_u, mean_col), inv_std_col)
                scan2_affine = pl.col_expand_add(pl.col_expand_mul(scan2_normalized, pl.cast(scan2_gamma, pl.FP32)), pl.cast(scan2_beta, pl.FP32))
                scan2_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan2_u, pl.BF16)
                pl.store(scan2_stored_h, [row_start, scan2_col], h)
                scan2_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan2_normalized, pl.BF16)
                pl.store(scan2_stored_xhat, [row_start, scan2_col], xhat)
                scan2_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan2_affine, pl.BF16)
                pl.store(scan2_stored_y, [row_start, scan2_col], y)
            if hidden % 256 != 0 and 0 < hidden:
                for scan3_col in pl.range(hidden // 256 * 256, hidden, 256):
                    scan3_cols = pl.min(256, hidden - scan3_col)
                    scan3_x = pl.load(x, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                    scan3_res = pl.load(res, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                    scan3_u = pl.add(pl.cast(scan3_x, pl.FP32), pl.cast(scan3_res, pl.FP32))
                    scan3_gamma = pl.load(gamma, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                    scan3_beta = pl.load(beta, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                    scan3_normalized = pl.row_expand_mul(pl.row_expand_sub(scan3_u, mean_col), inv_std_col)
                    scan3_affine = pl.col_expand_add(pl.col_expand_mul(scan3_normalized, pl.cast(scan3_gamma, pl.FP32)), pl.cast(scan3_beta, pl.FP32))
                    scan3_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan3_u, pl.BF16)
                    pl.store(pl.set_validshape(scan3_stored_h, 32, scan3_cols), [row_start, scan3_col], h)
                    scan3_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan3_normalized, pl.BF16)
                    pl.store(pl.set_validshape(scan3_stored_xhat, 32, scan3_cols), [row_start, scan3_col], xhat)
                    scan3_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan3_affine, pl.BF16)
                    pl.store(pl.set_validshape(scan3_stored_y, 32, scan3_cols), [row_start, scan3_col], y)
    if rows % 32 != 0:
        for row_start in pl.parallel(rows // 32 * 32, rows, 32):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='boundary_rows'):
                valid_rows = pl.min(32, rows - row_start)
                sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
                sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
                for scan4_col in pl.range(0, hidden // 128 * 128, 128):
                    scan4_x = pl.load(x, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                    scan4_res = pl.load(res, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                    scan4_u = pl.add(pl.cast(scan4_x, pl.FP32), pl.cast(scan4_res, pl.FP32))
                    scan4_u = pl.fillpad(pl.set_validshape(scan4_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                    scan4_u = pl.set_validshape(scan4_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan4_u)
                    sq_acc = pl.add(sq_acc, pl.mul(scan4_u, scan4_u))
                if hidden % 128 != 0 and 0 < hidden:
                    for scan5_col in pl.range(hidden // 128 * 128, hidden, 128):
                        scan5_cols = pl.min(128, hidden - scan5_col)
                        scan5_x = pl.load(x, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                        scan5_res = pl.load(res, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                        scan5_u = pl.add(pl.cast(scan5_x, pl.FP32), pl.cast(scan5_res, pl.FP32))
                        scan5_u = pl.fillpad(pl.set_validshape(scan5_u, valid_rows, scan5_cols), pad_value=pl.PadValue.zero)
                        scan5_u = pl.set_validshape(scan5_u, 32, 128)
                        sum_acc = pl.add(sum_acc, scan5_u)
                        sq_acc = pl.add(sq_acc, pl.mul(scan5_u, scan5_u))
                square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
                mean_square_row = pl.mul(square_row, 1.0 / hidden)
                sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
                mean_row = pl.mul(sum_row, 1.0 / hidden)
                mean_col = pl.reshape(mean_row, [32, 1])
                variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
                inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
                inv_std_col = pl.reshape(inv_std_row, [32, 1])
                pl.store(pl.set_validshape(mean_row, 1, valid_rows), [0, row_start], mu)
                pl.store(pl.set_validshape(inv_std_row, 1, valid_rows), [0, row_start], rstd)
                for scan6_col in pl.range(0, hidden // 256 * 256, 256):
                    scan6_x = pl.load(x, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                    scan6_res = pl.load(res, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                    scan6_u = pl.add(pl.cast(scan6_x, pl.FP32), pl.cast(scan6_res, pl.FP32))
                    scan6_gamma = pl.load(gamma, [0, scan6_col], [1, 256], valid_shape=[1, 256])
                    scan6_beta = pl.load(beta, [0, scan6_col], [1, 256], valid_shape=[1, 256])
                    scan6_normalized = pl.row_expand_mul(pl.row_expand_sub(scan6_u, mean_col), inv_std_col)
                    scan6_affine = pl.col_expand_add(pl.col_expand_mul(scan6_normalized, pl.cast(scan6_gamma, pl.FP32)), pl.cast(scan6_beta, pl.FP32))
                    scan6_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan6_u, pl.BF16)
                    pl.store(pl.set_validshape(scan6_stored_h, valid_rows, 256), [row_start, scan6_col], h)
                    scan6_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan6_normalized, pl.BF16)
                    pl.store(pl.set_validshape(scan6_stored_xhat, valid_rows, 256), [row_start, scan6_col], xhat)
                    scan6_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan6_affine, pl.BF16)
                    pl.store(pl.set_validshape(scan6_stored_y, valid_rows, 256), [row_start, scan6_col], y)
                if hidden % 256 != 0 and 0 < hidden:
                    for scan7_col in pl.range(hidden // 256 * 256, hidden, 256):
                        scan7_cols = pl.min(256, hidden - scan7_col)
                        scan7_x = pl.load(x, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                        scan7_res = pl.load(res, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                        scan7_u = pl.add(pl.cast(scan7_x, pl.FP32), pl.cast(scan7_res, pl.FP32))
                        scan7_gamma = pl.load(gamma, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                        scan7_beta = pl.load(beta, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                        scan7_normalized = pl.row_expand_mul(pl.row_expand_sub(scan7_u, mean_col), inv_std_col)
                        scan7_affine = pl.col_expand_add(pl.col_expand_mul(scan7_normalized, pl.cast(scan7_gamma, pl.FP32)), pl.cast(scan7_beta, pl.FP32))
                        scan7_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan7_u, pl.BF16)
                        pl.store(pl.set_validshape(scan7_stored_h, valid_rows, scan7_cols), [row_start, scan7_col], h)
                        scan7_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan7_normalized, pl.BF16)
                        pl.store(pl.set_validshape(scan7_stored_xhat, valid_rows, scan7_cols), [row_start, scan7_col], xhat)
                        scan7_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan7_affine, pl.BF16)
                        pl.store(pl.set_validshape(scan7_stored_y, valid_rows, scan7_cols), [row_start, scan7_col], y)
    return y


@pl.jit
def singleton_fp32(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="singleton_rows"):
            valid_rows = pl.min(ROW_TILE, rows - row_start)
            x_tile = pl.slice(x, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            res_tile = pl.slice(res, [ROW_TILE, 1], [row_start, 0], valid_shape=[valid_rows, 1])
            x_tile = pl.set_validshape(x_tile, ROW_TILE, 1)
            res_tile = pl.set_validshape(res_tile, ROW_TILE, 1)
            residual = pl.add(x_tile, res_tile)
            mean_row = pl.reshape(residual, [1, ROW_TILE])
            inv_std_row = pl.rsqrt(pl.full([1, ROW_TILE], dtype=pl.FP32, value=EPS), high_precision=True)
            normalized_row = pl.mul(pl.sub(mean_row, mean_row), 316.2277660168379)
            gamma_tile = pl.reshape(pl.slice(gamma, [1, 1], [0, 0]), [1, 1])
            beta_tile = pl.reshape(pl.slice(beta, [1, 1], [0, 0]), [1, 1])
            gamma_scalar = pl.read(gamma_tile, [0, 0])
            beta_scalar = pl.read(beta_tile, [0, 0])
            affine_row = pl.add(pl.mul(normalized_row, gamma_scalar), beta_scalar)
            normalized = pl.reshape(normalized_row, [ROW_TILE, 1])
            affine = pl.reshape(affine_row, [ROW_TILE, 1])
            h[row_start:row_start + ROW_TILE, 0:1] = pl.set_validshape(residual, valid_rows, 1)
            xhat[row_start:row_start + ROW_TILE, 0:1] = pl.set_validshape(normalized, valid_rows, 1)
            y[row_start:row_start + ROW_TILE, 0:1] = pl.set_validshape(affine, valid_rows, 1)
            mu[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + ROW_TILE] = pl.set_validshape(inv_std_row, 1, valid_rows)
    return y


@pl.jit
def peeled_fp32(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows // 32 * 32, 32):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="interior_rows"):
            sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan0_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan0_x = pl.slice(x, [32, 128], [row_start, scan0_col], valid_shape=[32, 128])
                scan0_res = pl.slice(res, [32, 128], [row_start, scan0_col], valid_shape=[32, 128])
                scan0_u = pl.add(scan0_x, scan0_res)
                sum_acc = pl.add(sum_acc, scan0_u)
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan1_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan1_cols = pl.min(128, (hidden) - scan1_col)
                    scan1_x = pl.slice(x, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                    scan1_res = pl.slice(res, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                    scan1_u = pl.add(scan1_x, scan1_res)
                    scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                    scan1_u = pl.set_validshape(scan1_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan1_u)
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan2_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan2_x = pl.slice(x, [32, 128], [row_start, scan2_col], valid_shape=[32, 128])
                scan2_res = pl.slice(res, [32, 128], [row_start, scan2_col], valid_shape=[32, 128])
                scan2_u = pl.add(scan2_x, scan2_res)
                scan2_centred = pl.row_expand_sub(scan2_u, mean_col)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan2_centred, scan2_centred))
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan3_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan3_cols = pl.min(128, (hidden) - scan3_col)
                    scan3_x = pl.slice(x, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                    scan3_res = pl.slice(res, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                    scan3_u = pl.add(scan3_x, scan3_res)
                    scan3_centred = pl.row_expand_sub(scan3_u, mean_col)
                    scan3_centred = pl.fillpad(pl.set_validshape(scan3_centred, 32, scan3_cols), pad_value=pl.PadValue.zero)
                    scan3_centred = pl.set_validshape(scan3_centred, 32, 128)
                    centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan3_centred, scan3_centred))
            centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
            variance_row = pl.mul(centred_square_row, 1.0 / hidden)
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, 32)
            rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, 32)
            for scan4_col in pl.range(0, (hidden) // 256 * 256, 256):
                scan4_x = pl.slice(x, [32, 256], [row_start, scan4_col], valid_shape=[32, 256])
                scan4_res = pl.slice(res, [32, 256], [row_start, scan4_col], valid_shape=[32, 256])
                scan4_u = pl.add(scan4_x, scan4_res)
                scan4_gamma = pl.slice(gamma, [1, 256], [0, scan4_col], valid_shape=[1, 256])
                scan4_beta = pl.slice(beta, [1, 256], [0, scan4_col], valid_shape=[1, 256])
                scan4_normalized = pl.row_expand_mul(pl.row_expand_sub(scan4_u, mean_col), inv_std_col)
                scan4_affine = pl.col_expand_add(pl.col_expand_mul(scan4_normalized, scan4_gamma), scan4_beta)
                h[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_u
                xhat[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_normalized
                y[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_affine
            if (hidden) % 256 != 0 and (0) < (hidden):
                for scan5_col in pl.range((hidden) // 256 * 256, hidden, 256):
                    scan5_cols = pl.min(256, (hidden) - scan5_col)
                    scan5_x = pl.slice(x, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                    scan5_res = pl.slice(res, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                    scan5_u = pl.add(scan5_x, scan5_res)
                    scan5_gamma = pl.slice(gamma, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                    scan5_beta = pl.slice(beta, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                    scan5_normalized = pl.row_expand_mul(pl.row_expand_sub(scan5_u, mean_col), inv_std_col)
                    scan5_affine = pl.col_expand_add(pl.col_expand_mul(scan5_normalized, scan5_gamma), scan5_beta)
                    h[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_u, 32, scan5_cols)
                    xhat[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_normalized, 32, scan5_cols)
                    y[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_affine, 32, scan5_cols)
    if rows % 32 != 0:
        for row_start in pl.parallel(rows // 32 * 32, rows, 32):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="boundary_rows"):
                valid_rows = pl.min(32, rows - row_start)
                sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
                for scan6_col in pl.range(0, (hidden) // 128 * 128, 128):
                    scan6_x = pl.slice(x, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                    scan6_res = pl.slice(res, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                    scan6_u = pl.add(scan6_x, scan6_res)
                    scan6_u = pl.fillpad(pl.set_validshape(scan6_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                    scan6_u = pl.set_validshape(scan6_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan6_u)
                if (hidden) % 128 != 0 and (0) < (hidden):
                    for scan7_col in pl.range((hidden) // 128 * 128, hidden, 128):
                        scan7_cols = pl.min(128, (hidden) - scan7_col)
                        scan7_x = pl.slice(x, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                        scan7_res = pl.slice(res, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                        scan7_u = pl.add(scan7_x, scan7_res)
                        scan7_u = pl.fillpad(pl.set_validshape(scan7_u, valid_rows, scan7_cols), pad_value=pl.PadValue.zero)
                        scan7_u = pl.set_validshape(scan7_u, 32, 128)
                        sum_acc = pl.add(sum_acc, scan7_u)
                sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
                mean_row = pl.mul(sum_row, 1.0 / hidden)
                mean_col = pl.reshape(mean_row, [32, 1])
                centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
                for scan8_col in pl.range(0, (hidden) // 128 * 128, 128):
                    scan8_x = pl.slice(x, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                    scan8_res = pl.slice(res, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                    scan8_u = pl.add(scan8_x, scan8_res)
                    scan8_centred = pl.row_expand_sub(scan8_u, mean_col)
                    scan8_centred = pl.fillpad(pl.set_validshape(scan8_centred, valid_rows, 128), pad_value=pl.PadValue.zero)
                    scan8_centred = pl.set_validshape(scan8_centred, 32, 128)
                    centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan8_centred, scan8_centred))
                if (hidden) % 128 != 0 and (0) < (hidden):
                    for scan9_col in pl.range((hidden) // 128 * 128, hidden, 128):
                        scan9_cols = pl.min(128, (hidden) - scan9_col)
                        scan9_x = pl.slice(x, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                        scan9_res = pl.slice(res, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                        scan9_u = pl.add(scan9_x, scan9_res)
                        scan9_centred = pl.row_expand_sub(scan9_u, mean_col)
                        scan9_centred = pl.fillpad(pl.set_validshape(scan9_centred, valid_rows, scan9_cols), pad_value=pl.PadValue.zero)
                        scan9_centred = pl.set_validshape(scan9_centred, 32, 128)
                        centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan9_centred, scan9_centred))
                centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
                variance_row = pl.mul(centred_square_row, 1.0 / hidden)
                inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
                inv_std_col = pl.reshape(inv_std_row, [32, 1])
                mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, valid_rows)
                rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, valid_rows)
                for scan10_col in pl.range(0, (hidden) // 256 * 256, 256):
                    scan10_x = pl.slice(x, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                    scan10_res = pl.slice(res, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                    scan10_u = pl.add(scan10_x, scan10_res)
                    scan10_gamma = pl.slice(gamma, [1, 256], [0, scan10_col], valid_shape=[1, 256])
                    scan10_beta = pl.slice(beta, [1, 256], [0, scan10_col], valid_shape=[1, 256])
                    scan10_normalized = pl.row_expand_mul(pl.row_expand_sub(scan10_u, mean_col), inv_std_col)
                    scan10_affine = pl.col_expand_add(pl.col_expand_mul(scan10_normalized, scan10_gamma), scan10_beta)
                    h[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_u, valid_rows, 256)
                    xhat[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_normalized, valid_rows, 256)
                    y[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_affine, valid_rows, 256)
                if (hidden) % 256 != 0 and (0) < (hidden):
                    for scan11_col in pl.range((hidden) // 256 * 256, hidden, 256):
                        scan11_cols = pl.min(256, (hidden) - scan11_col)
                        scan11_x = pl.slice(x, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                        scan11_res = pl.slice(res, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                        scan11_u = pl.add(scan11_x, scan11_res)
                        scan11_gamma = pl.slice(gamma, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                        scan11_beta = pl.slice(beta, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                        scan11_normalized = pl.row_expand_mul(pl.row_expand_sub(scan11_u, mean_col), inv_std_col)
                        scan11_affine = pl.col_expand_add(pl.col_expand_mul(scan11_normalized, scan11_gamma), scan11_beta)
                        h[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_u, valid_rows, scan11_cols)
                        xhat[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_normalized, valid_rows, scan11_cols)
                        y[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_affine, valid_rows, scan11_cols)
    return y


@pl.jit
def vector_singleton_bf16(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    flat_x = pl.reshape(x, [1, rows])
    flat_res = pl.reshape(res, [1, rows])
    flat_h = pl.reshape(h, [1, rows])
    flat_xhat = pl.reshape(xhat, [1, rows])
    flat_y = pl.reshape(y, [1, rows])
    for start in pl.parallel(0, rows, 256):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="singleton_vector"):
            count = pl.min(256, rows - start)
            if rows == 1:
                x_scalar = pl.read(pl.cast(pl.slice(x, [16, 1], [0, 0], valid_shape=[1, 1]), pl.FP32), [0, 0])
                res_scalar = pl.read(pl.cast(pl.slice(res, [16, 1], [0, 0], valid_shape=[1, 1]), pl.FP32), [0, 0])
                residual = pl.add(pl.full([1, 256], dtype=pl.FP32, value=0.0), x_scalar + res_scalar)
            else:
                x_tile = pl.slice(flat_x, [1, 256], [0, start], valid_shape=[1, count])
                res_tile = pl.slice(flat_res, [1, 256], [0, start], valid_shape=[1, count])
                residual = pl.add(pl.cast(x_tile, pl.FP32), pl.cast(res_tile, pl.FP32))
            gamma_scalar = pl.read(pl.cast(pl.slice(gamma, [16, 1], [0, 0], valid_shape=[1, 1]), pl.FP32), [0, 0])
            beta_scalar = pl.read(pl.cast(pl.slice(beta, [16, 1], [0, 0], valid_shape=[1, 1]), pl.FP32), [0, 0])
            normalized = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            affine = pl.add(pl.mul(normalized, gamma_scalar), beta_scalar)
            inv_std = pl.rsqrt(pl.full([1, 256], dtype=pl.FP32, value=EPS), high_precision=True)
            if rows == 1:
                pl.write(h, [0, 0], pl.read(pl.cast(residual, pl.BF16), [0, 0]))
                pl.write(xhat, [0, 0], pl.read(pl.cast(normalized, pl.BF16), [0, 0]))
                pl.write(y, [0, 0], pl.read(pl.cast(affine, pl.BF16), [0, 0]))
                pl.write(mu, [0, 0], pl.read(residual, [0, 0]))
                pl.write(rstd, [0, 0], pl.read(inv_std, [0, 0]))
            else:
                flat_h[0:1, start:start + 256] = pl.set_validshape(pl.cast(residual, pl.BF16), 1, count)
                flat_xhat[0:1, start:start + 256] = pl.set_validshape(pl.cast(normalized, pl.BF16), 1, count)
                flat_y[0:1, start:start + 256] = pl.set_validshape(pl.cast(affine, pl.BF16), 1, count)
                mu[0:1, start:start + 256] = pl.set_validshape(residual, 1, count)
                rstd[0:1, start:start + 256] = pl.set_validshape(inv_std, 1, count)
    return pl.reshape(flat_y, [rows, 1])


@pl.jit
def static_bf16(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows // 32 * 32, 32):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='interior_rows'):
            sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan0_col in pl.range(0, hidden // 128 * 128, 128):
                scan0_x = pl.load(x, [row_start, scan0_col], [32, 128])
                scan0_res = pl.load(res, [row_start, scan0_col], [32, 128])
                scan0_u = pl.add(pl.cast(scan0_x, pl.FP32), pl.cast(scan0_res, pl.FP32))
                sum_acc = pl.add(sum_acc, scan0_u)
                sq_acc = pl.add(sq_acc, pl.mul(scan0_u, scan0_u))
            if hidden % 128 != 0 and 0 < hidden:
                for scan1_col in pl.range(hidden // 128 * 128, hidden, 128):
                    scan1_cols = pl.min(128, hidden - scan1_col)
                    scan1_x = pl.load(x, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                    scan1_res = pl.load(res, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                    scan1_u = pl.add(pl.cast(scan1_x, pl.FP32), pl.cast(scan1_res, pl.FP32))
                    scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                    scan1_u = pl.set_validshape(scan1_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan1_u)
                    sq_acc = pl.add(sq_acc, pl.mul(scan1_u, scan1_u))
            square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_square_row = pl.mul(square_row, 1.0 / hidden)
            sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
            inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            pl.store(pl.set_validshape(mean_row, 1, 32), [0, row_start], mu)
            pl.store(pl.set_validshape(inv_std_row, 1, 32), [0, row_start], rstd)
            for scan2_col in pl.range(0, hidden // 256 * 256, 256):
                scan2_x = pl.load(x, [row_start, scan2_col], [32, 256])
                scan2_res = pl.load(res, [row_start, scan2_col], [32, 256])
                scan2_u = pl.add(pl.cast(scan2_x, pl.FP32), pl.cast(scan2_res, pl.FP32))
                scan2_gamma = pl.load(gamma, [0, scan2_col], [1, 256])
                scan2_beta = pl.load(beta, [0, scan2_col], [1, 256])
                scan2_normalized = pl.row_expand_mul(pl.row_expand_sub(scan2_u, mean_col), inv_std_col)
                scan2_affine = pl.col_expand_add(pl.col_expand_mul(scan2_normalized, pl.cast(scan2_gamma, pl.FP32)), pl.cast(scan2_beta, pl.FP32))
                scan2_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan2_u, pl.BF16)
                pl.store(scan2_stored_h, [row_start, scan2_col], h)
                scan2_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan2_normalized, pl.BF16)
                pl.store(scan2_stored_xhat, [row_start, scan2_col], xhat)
                scan2_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan2_affine, pl.BF16)
                pl.store(scan2_stored_y, [row_start, scan2_col], y)
            if hidden % 256 != 0 and 0 < hidden:
                for scan3_col in pl.range(hidden // 256 * 256, hidden, 256):
                    scan3_cols = pl.min(256, hidden - scan3_col)
                    scan3_x = pl.load(x, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                    scan3_res = pl.load(res, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                    scan3_u = pl.add(pl.cast(scan3_x, pl.FP32), pl.cast(scan3_res, pl.FP32))
                    scan3_gamma = pl.load(gamma, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                    scan3_beta = pl.load(beta, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                    scan3_normalized = pl.row_expand_mul(pl.row_expand_sub(scan3_u, mean_col), inv_std_col)
                    scan3_affine = pl.col_expand_add(pl.col_expand_mul(scan3_normalized, pl.cast(scan3_gamma, pl.FP32)), pl.cast(scan3_beta, pl.FP32))
                    scan3_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan3_u, pl.BF16)
                    pl.store(pl.set_validshape(scan3_stored_h, 32, scan3_cols), [row_start, scan3_col], h)
                    scan3_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan3_normalized, pl.BF16)
                    pl.store(pl.set_validshape(scan3_stored_xhat, 32, scan3_cols), [row_start, scan3_col], xhat)
                    scan3_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan3_affine, pl.BF16)
                    pl.store(pl.set_validshape(scan3_stored_y, 32, scan3_cols), [row_start, scan3_col], y)
    if rows % 32 != 0:
        for row_start in pl.parallel(rows // 32 * 32, rows, 32):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='boundary_rows'):
                valid_rows = pl.min(32, rows - row_start)
                sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
                sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
                for scan4_col in pl.range(0, hidden // 128 * 128, 128):
                    scan4_x = pl.load(x, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                    scan4_res = pl.load(res, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                    scan4_u = pl.add(pl.cast(scan4_x, pl.FP32), pl.cast(scan4_res, pl.FP32))
                    scan4_u = pl.fillpad(pl.set_validshape(scan4_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                    scan4_u = pl.set_validshape(scan4_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan4_u)
                    sq_acc = pl.add(sq_acc, pl.mul(scan4_u, scan4_u))
                if hidden % 128 != 0 and 0 < hidden:
                    for scan5_col in pl.range(hidden // 128 * 128, hidden, 128):
                        scan5_cols = pl.min(128, hidden - scan5_col)
                        scan5_x = pl.load(x, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                        scan5_res = pl.load(res, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                        scan5_u = pl.add(pl.cast(scan5_x, pl.FP32), pl.cast(scan5_res, pl.FP32))
                        scan5_u = pl.fillpad(pl.set_validshape(scan5_u, valid_rows, scan5_cols), pad_value=pl.PadValue.zero)
                        scan5_u = pl.set_validshape(scan5_u, 32, 128)
                        sum_acc = pl.add(sum_acc, scan5_u)
                        sq_acc = pl.add(sq_acc, pl.mul(scan5_u, scan5_u))
                square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
                mean_square_row = pl.mul(square_row, 1.0 / hidden)
                sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
                mean_row = pl.mul(sum_row, 1.0 / hidden)
                mean_col = pl.reshape(mean_row, [32, 1])
                variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
                inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
                inv_std_col = pl.reshape(inv_std_row, [32, 1])
                pl.store(pl.set_validshape(mean_row, 1, valid_rows), [0, row_start], mu)
                pl.store(pl.set_validshape(inv_std_row, 1, valid_rows), [0, row_start], rstd)
                for scan6_col in pl.range(0, hidden // 256 * 256, 256):
                    scan6_x = pl.load(x, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                    scan6_res = pl.load(res, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                    scan6_u = pl.add(pl.cast(scan6_x, pl.FP32), pl.cast(scan6_res, pl.FP32))
                    scan6_gamma = pl.load(gamma, [0, scan6_col], [1, 256])
                    scan6_beta = pl.load(beta, [0, scan6_col], [1, 256])
                    scan6_normalized = pl.row_expand_mul(pl.row_expand_sub(scan6_u, mean_col), inv_std_col)
                    scan6_affine = pl.col_expand_add(pl.col_expand_mul(scan6_normalized, pl.cast(scan6_gamma, pl.FP32)), pl.cast(scan6_beta, pl.FP32))
                    scan6_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan6_u, pl.BF16)
                    pl.store(pl.set_validshape(scan6_stored_h, valid_rows, 256), [row_start, scan6_col], h)
                    scan6_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan6_normalized, pl.BF16)
                    pl.store(pl.set_validshape(scan6_stored_xhat, valid_rows, 256), [row_start, scan6_col], xhat)
                    scan6_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan6_affine, pl.BF16)
                    pl.store(pl.set_validshape(scan6_stored_y, valid_rows, 256), [row_start, scan6_col], y)
                if hidden % 256 != 0 and 0 < hidden:
                    for scan7_col in pl.range(hidden // 256 * 256, hidden, 256):
                        scan7_cols = pl.min(256, hidden - scan7_col)
                        scan7_x = pl.load(x, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                        scan7_res = pl.load(res, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                        scan7_u = pl.add(pl.cast(scan7_x, pl.FP32), pl.cast(scan7_res, pl.FP32))
                        scan7_gamma = pl.load(gamma, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                        scan7_beta = pl.load(beta, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                        scan7_normalized = pl.row_expand_mul(pl.row_expand_sub(scan7_u, mean_col), inv_std_col)
                        scan7_affine = pl.col_expand_add(pl.col_expand_mul(scan7_normalized, pl.cast(scan7_gamma, pl.FP32)), pl.cast(scan7_beta, pl.FP32))
                        scan7_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan7_u, pl.BF16)
                        pl.store(pl.set_validshape(scan7_stored_h, valid_rows, scan7_cols), [row_start, scan7_col], h)
                        scan7_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan7_normalized, pl.BF16)
                        pl.store(pl.set_validshape(scan7_stored_xhat, valid_rows, scan7_cols), [row_start, scan7_col], xhat)
                        scan7_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan7_affine, pl.BF16)
                        pl.store(pl.set_validshape(scan7_stored_y, valid_rows, scan7_cols), [row_start, scan7_col], y)
    return y


@pl.jit
def vector_singleton_fp32(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    flat_x = pl.reshape(x, [1, rows])
    flat_res = pl.reshape(res, [1, rows])
    flat_h = pl.reshape(h, [1, rows])
    flat_xhat = pl.reshape(xhat, [1, rows])
    flat_y = pl.reshape(y, [1, rows])
    for start in pl.parallel(0, rows, 256):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="singleton_vector"):
            count = pl.min(256, rows - start)
            if rows == 1:
                x_scalar = pl.read(x, [0, 0])
                res_scalar = pl.read(res, [0, 0])
                residual = pl.add(pl.full([1, 256], dtype=pl.FP32, value=0.0), x_scalar + res_scalar)
            else:
                x_tile = pl.slice(flat_x, [1, 256], [0, start], valid_shape=[1, count])
                res_tile = pl.slice(flat_res, [1, 256], [0, start], valid_shape=[1, count])
                residual = pl.add(x_tile, res_tile)
            gamma_scalar = pl.read(gamma, [0, 0])
            beta_scalar = pl.read(beta, [0, 0])
            normalized = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            affine = pl.add(pl.mul(normalized, gamma_scalar), beta_scalar)
            inv_std = pl.rsqrt(pl.full([1, 256], dtype=pl.FP32, value=EPS), high_precision=True)
            if rows == 1:
                pl.write(h, [0, 0], pl.read(residual, [0, 0]))
                pl.write(xhat, [0, 0], pl.read(normalized, [0, 0]))
                pl.write(y, [0, 0], pl.read(affine, [0, 0]))
                pl.write(mu, [0, 0], pl.read(residual, [0, 0]))
                pl.write(rstd, [0, 0], pl.read(inv_std, [0, 0]))
            else:
                flat_h[0:1, start:start + 256] = pl.set_validshape(residual, 1, count)
                flat_xhat[0:1, start:start + 256] = pl.set_validshape(normalized, 1, count)
                flat_y[0:1, start:start + 256] = pl.set_validshape(affine, 1, count)
                mu[0:1, start:start + 256] = pl.set_validshape(residual, 1, count)
                rstd[0:1, start:start + 256] = pl.set_validshape(inv_std, 1, count)
    return pl.reshape(flat_y, [rows, 1])


@pl.jit
def static_fp32(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows // 32 * 32, 32):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="interior_rows"):
            sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan0_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan0_x = pl.slice(x, [32, 128], [row_start, scan0_col])
                scan0_res = pl.slice(res, [32, 128], [row_start, scan0_col])
                scan0_u = pl.add(scan0_x, scan0_res)
                sum_acc = pl.add(sum_acc, scan0_u)
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan1_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan1_cols = pl.min(128, (hidden) - scan1_col)
                    scan1_x = pl.slice(x, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                    scan1_res = pl.slice(res, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                    scan1_u = pl.add(scan1_x, scan1_res)
                    scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                    scan1_u = pl.set_validshape(scan1_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan1_u)
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan2_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan2_x = pl.slice(x, [32, 128], [row_start, scan2_col])
                scan2_res = pl.slice(res, [32, 128], [row_start, scan2_col])
                scan2_u = pl.add(scan2_x, scan2_res)
                scan2_centred = pl.row_expand_sub(scan2_u, mean_col)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan2_centred, scan2_centred))
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan3_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan3_cols = pl.min(128, (hidden) - scan3_col)
                    scan3_x = pl.slice(x, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                    scan3_res = pl.slice(res, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                    scan3_u = pl.add(scan3_x, scan3_res)
                    scan3_centred = pl.row_expand_sub(scan3_u, mean_col)
                    scan3_centred = pl.fillpad(pl.set_validshape(scan3_centred, 32, scan3_cols), pad_value=pl.PadValue.zero)
                    scan3_centred = pl.set_validshape(scan3_centred, 32, 128)
                    centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan3_centred, scan3_centred))
            centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
            variance_row = pl.mul(centred_square_row, 1.0 / hidden)
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, 32)
            rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, 32)
            for scan4_col in pl.range(0, (hidden) // 256 * 256, 256):
                scan4_x = pl.slice(x, [32, 256], [row_start, scan4_col])
                scan4_res = pl.slice(res, [32, 256], [row_start, scan4_col])
                scan4_u = pl.add(scan4_x, scan4_res)
                scan4_gamma = pl.slice(gamma, [1, 256], [0, scan4_col])
                scan4_beta = pl.slice(beta, [1, 256], [0, scan4_col])
                scan4_normalized = pl.row_expand_mul(pl.row_expand_sub(scan4_u, mean_col), inv_std_col)
                scan4_affine = pl.col_expand_add(pl.col_expand_mul(scan4_normalized, scan4_gamma), scan4_beta)
                h[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_u
                xhat[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_normalized
                y[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_affine
            if (hidden) % 256 != 0 and (0) < (hidden):
                for scan5_col in pl.range((hidden) // 256 * 256, hidden, 256):
                    scan5_cols = pl.min(256, (hidden) - scan5_col)
                    scan5_x = pl.slice(x, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                    scan5_res = pl.slice(res, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                    scan5_u = pl.add(scan5_x, scan5_res)
                    scan5_gamma = pl.slice(gamma, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                    scan5_beta = pl.slice(beta, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                    scan5_normalized = pl.row_expand_mul(pl.row_expand_sub(scan5_u, mean_col), inv_std_col)
                    scan5_affine = pl.col_expand_add(pl.col_expand_mul(scan5_normalized, scan5_gamma), scan5_beta)
                    h[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_u, 32, scan5_cols)
                    xhat[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_normalized, 32, scan5_cols)
                    y[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_affine, 32, scan5_cols)
    if rows % 32 != 0:
        for row_start in pl.parallel(rows // 32 * 32, rows, 32):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="boundary_rows"):
                valid_rows = pl.min(32, rows - row_start)
                sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
                for scan6_col in pl.range(0, (hidden) // 128 * 128, 128):
                    scan6_x = pl.slice(x, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                    scan6_res = pl.slice(res, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                    scan6_u = pl.add(scan6_x, scan6_res)
                    scan6_u = pl.fillpad(pl.set_validshape(scan6_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                    scan6_u = pl.set_validshape(scan6_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan6_u)
                if (hidden) % 128 != 0 and (0) < (hidden):
                    for scan7_col in pl.range((hidden) // 128 * 128, hidden, 128):
                        scan7_cols = pl.min(128, (hidden) - scan7_col)
                        scan7_x = pl.slice(x, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                        scan7_res = pl.slice(res, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                        scan7_u = pl.add(scan7_x, scan7_res)
                        scan7_u = pl.fillpad(pl.set_validshape(scan7_u, valid_rows, scan7_cols), pad_value=pl.PadValue.zero)
                        scan7_u = pl.set_validshape(scan7_u, 32, 128)
                        sum_acc = pl.add(sum_acc, scan7_u)
                sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
                mean_row = pl.mul(sum_row, 1.0 / hidden)
                mean_col = pl.reshape(mean_row, [32, 1])
                centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
                for scan8_col in pl.range(0, (hidden) // 128 * 128, 128):
                    scan8_x = pl.slice(x, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                    scan8_res = pl.slice(res, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                    scan8_u = pl.add(scan8_x, scan8_res)
                    scan8_centred = pl.row_expand_sub(scan8_u, mean_col)
                    scan8_centred = pl.fillpad(pl.set_validshape(scan8_centred, valid_rows, 128), pad_value=pl.PadValue.zero)
                    scan8_centred = pl.set_validshape(scan8_centred, 32, 128)
                    centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan8_centred, scan8_centred))
                if (hidden) % 128 != 0 and (0) < (hidden):
                    for scan9_col in pl.range((hidden) // 128 * 128, hidden, 128):
                        scan9_cols = pl.min(128, (hidden) - scan9_col)
                        scan9_x = pl.slice(x, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                        scan9_res = pl.slice(res, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                        scan9_u = pl.add(scan9_x, scan9_res)
                        scan9_centred = pl.row_expand_sub(scan9_u, mean_col)
                        scan9_centred = pl.fillpad(pl.set_validshape(scan9_centred, valid_rows, scan9_cols), pad_value=pl.PadValue.zero)
                        scan9_centred = pl.set_validshape(scan9_centred, 32, 128)
                        centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan9_centred, scan9_centred))
                centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
                variance_row = pl.mul(centred_square_row, 1.0 / hidden)
                inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
                inv_std_col = pl.reshape(inv_std_row, [32, 1])
                mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, valid_rows)
                rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, valid_rows)
                for scan10_col in pl.range(0, (hidden) // 256 * 256, 256):
                    scan10_x = pl.slice(x, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                    scan10_res = pl.slice(res, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                    scan10_u = pl.add(scan10_x, scan10_res)
                    scan10_gamma = pl.slice(gamma, [1, 256], [0, scan10_col])
                    scan10_beta = pl.slice(beta, [1, 256], [0, scan10_col])
                    scan10_normalized = pl.row_expand_mul(pl.row_expand_sub(scan10_u, mean_col), inv_std_col)
                    scan10_affine = pl.col_expand_add(pl.col_expand_mul(scan10_normalized, scan10_gamma), scan10_beta)
                    h[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_u, valid_rows, 256)
                    xhat[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_normalized, valid_rows, 256)
                    y[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_affine, valid_rows, 256)
                if (hidden) % 256 != 0 and (0) < (hidden):
                    for scan11_col in pl.range((hidden) // 256 * 256, hidden, 256):
                        scan11_cols = pl.min(256, (hidden) - scan11_col)
                        scan11_x = pl.slice(x, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                        scan11_res = pl.slice(res, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                        scan11_u = pl.add(scan11_x, scan11_res)
                        scan11_gamma = pl.slice(gamma, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                        scan11_beta = pl.slice(beta, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                        scan11_normalized = pl.row_expand_mul(pl.row_expand_sub(scan11_u, mean_col), inv_std_col)
                        scan11_affine = pl.col_expand_add(pl.col_expand_mul(scan11_normalized, scan11_gamma), scan11_beta)
                        h[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_u, valid_rows, scan11_cols)
                        xhat[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_normalized, valid_rows, scan11_cols)
                        y[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_affine, valid_rows, scan11_cols)
    return y


@pl.jit
def scheduled_peeled_bf16_1(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_group_0 in pl.spmd((rows // 32 * 32 - 0 + 31) // 32, name_hint='row_fanout_0', sync_start=False):
        row_start = 0 + row_group_0 * 32
        sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
        sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
        for scan0_col in pl.range(0, hidden // 128 * 128, 128):
            scan0_x = pl.load(x, [row_start, scan0_col], [32, 128], valid_shape=[32, 128])
            scan0_res = pl.load(res, [row_start, scan0_col], [32, 128], valid_shape=[32, 128])
            scan0_u = pl.add(pl.cast(scan0_x, pl.FP32), pl.cast(scan0_res, pl.FP32))
            sum_acc = pl.add(sum_acc, scan0_u)
            sq_acc = pl.add(sq_acc, pl.mul(scan0_u, scan0_u))
        if hidden % 128 != 0 and 0 < hidden:
            for scan1_col in pl.range(hidden // 128 * 128, hidden, 128):
                scan1_cols = pl.min(128, hidden - scan1_col)
                scan1_x = pl.load(x, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                scan1_res = pl.load(res, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                scan1_u = pl.add(pl.cast(scan1_x, pl.FP32), pl.cast(scan1_res, pl.FP32))
                scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                scan1_u = pl.set_validshape(scan1_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan1_u)
                sq_acc = pl.add(sq_acc, pl.mul(scan1_u, scan1_u))
        square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
        mean_square_row = pl.mul(square_row, 1.0 / hidden)
        sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
        mean_row = pl.mul(sum_row, 1.0 / hidden)
        mean_col = pl.reshape(mean_row, [32, 1])
        variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
        inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
        inv_std_col = pl.reshape(inv_std_row, [32, 1])
        pl.store(pl.set_validshape(mean_row, 1, 32), [0, row_start], mu)
        pl.store(pl.set_validshape(inv_std_row, 1, 32), [0, row_start], rstd)
        for scan2_col in pl.range(0, hidden // 256 * 256, 256):
            scan2_x = pl.load(x, [row_start, scan2_col], [32, 256], valid_shape=[32, 256])
            scan2_res = pl.load(res, [row_start, scan2_col], [32, 256], valid_shape=[32, 256])
            scan2_u = pl.add(pl.cast(scan2_x, pl.FP32), pl.cast(scan2_res, pl.FP32))
            scan2_gamma = pl.load(gamma, [0, scan2_col], [1, 256], valid_shape=[1, 256])
            scan2_beta = pl.load(beta, [0, scan2_col], [1, 256], valid_shape=[1, 256])
            scan2_normalized = pl.row_expand_mul(pl.row_expand_sub(scan2_u, mean_col), inv_std_col)
            scan2_affine = pl.col_expand_add(pl.col_expand_mul(scan2_normalized, pl.cast(scan2_gamma, pl.FP32)), pl.cast(scan2_beta, pl.FP32))
            scan2_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan2_u, pl.BF16)
            pl.store(scan2_stored_h, [row_start, scan2_col], h)
            scan2_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan2_normalized, pl.BF16)
            pl.store(scan2_stored_xhat, [row_start, scan2_col], xhat)
            scan2_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan2_affine, pl.BF16)
            pl.store(scan2_stored_y, [row_start, scan2_col], y)
        if hidden % 256 != 0 and 0 < hidden:
            for scan3_col in pl.range(hidden // 256 * 256, hidden, 256):
                scan3_cols = pl.min(256, hidden - scan3_col)
                scan3_x = pl.load(x, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                scan3_res = pl.load(res, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                scan3_u = pl.add(pl.cast(scan3_x, pl.FP32), pl.cast(scan3_res, pl.FP32))
                scan3_gamma = pl.load(gamma, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                scan3_beta = pl.load(beta, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                scan3_normalized = pl.row_expand_mul(pl.row_expand_sub(scan3_u, mean_col), inv_std_col)
                scan3_affine = pl.col_expand_add(pl.col_expand_mul(scan3_normalized, pl.cast(scan3_gamma, pl.FP32)), pl.cast(scan3_beta, pl.FP32))
                scan3_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan3_u, pl.BF16)
                pl.store(pl.set_validshape(scan3_stored_h, 32, scan3_cols), [row_start, scan3_col], h)
                scan3_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan3_normalized, pl.BF16)
                pl.store(pl.set_validshape(scan3_stored_xhat, 32, scan3_cols), [row_start, scan3_col], xhat)
                scan3_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan3_affine, pl.BF16)
                pl.store(pl.set_validshape(scan3_stored_y, 32, scan3_cols), [row_start, scan3_col], y)
    if rows % 32 != 0:
        for row_group_1 in pl.spmd(1, name_hint='row_fanout_1', sync_start=False):
            row_start = rows // 32 * 32 + row_group_1 * 32
            valid_rows = pl.min(32, rows - row_start)
            sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan4_col in pl.range(0, hidden // 128 * 128, 128):
                scan4_x = pl.load(x, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                scan4_res = pl.load(res, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                scan4_u = pl.add(pl.cast(scan4_x, pl.FP32), pl.cast(scan4_res, pl.FP32))
                scan4_u = pl.fillpad(pl.set_validshape(scan4_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                scan4_u = pl.set_validshape(scan4_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan4_u)
                sq_acc = pl.add(sq_acc, pl.mul(scan4_u, scan4_u))
            if hidden % 128 != 0 and 0 < hidden:
                for scan5_col in pl.range(hidden // 128 * 128, hidden, 128):
                    scan5_cols = pl.min(128, hidden - scan5_col)
                    scan5_x = pl.load(x, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                    scan5_res = pl.load(res, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                    scan5_u = pl.add(pl.cast(scan5_x, pl.FP32), pl.cast(scan5_res, pl.FP32))
                    scan5_u = pl.fillpad(pl.set_validshape(scan5_u, valid_rows, scan5_cols), pad_value=pl.PadValue.zero)
                    scan5_u = pl.set_validshape(scan5_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan5_u)
                    sq_acc = pl.add(sq_acc, pl.mul(scan5_u, scan5_u))
            square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_square_row = pl.mul(square_row, 1.0 / hidden)
            sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
            inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            pl.store(pl.set_validshape(mean_row, 1, valid_rows), [0, row_start], mu)
            pl.store(pl.set_validshape(inv_std_row, 1, valid_rows), [0, row_start], rstd)
            for scan6_col in pl.range(0, hidden // 256 * 256, 256):
                scan6_x = pl.load(x, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                scan6_res = pl.load(res, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                scan6_u = pl.add(pl.cast(scan6_x, pl.FP32), pl.cast(scan6_res, pl.FP32))
                scan6_gamma = pl.load(gamma, [0, scan6_col], [1, 256], valid_shape=[1, 256])
                scan6_beta = pl.load(beta, [0, scan6_col], [1, 256], valid_shape=[1, 256])
                scan6_normalized = pl.row_expand_mul(pl.row_expand_sub(scan6_u, mean_col), inv_std_col)
                scan6_affine = pl.col_expand_add(pl.col_expand_mul(scan6_normalized, pl.cast(scan6_gamma, pl.FP32)), pl.cast(scan6_beta, pl.FP32))
                scan6_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan6_u, pl.BF16)
                pl.store(pl.set_validshape(scan6_stored_h, valid_rows, 256), [row_start, scan6_col], h)
                scan6_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan6_normalized, pl.BF16)
                pl.store(pl.set_validshape(scan6_stored_xhat, valid_rows, 256), [row_start, scan6_col], xhat)
                scan6_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan6_affine, pl.BF16)
                pl.store(pl.set_validshape(scan6_stored_y, valid_rows, 256), [row_start, scan6_col], y)
            if hidden % 256 != 0 and 0 < hidden:
                for scan7_col in pl.range(hidden // 256 * 256, hidden, 256):
                    scan7_cols = pl.min(256, hidden - scan7_col)
                    scan7_x = pl.load(x, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                    scan7_res = pl.load(res, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                    scan7_u = pl.add(pl.cast(scan7_x, pl.FP32), pl.cast(scan7_res, pl.FP32))
                    scan7_gamma = pl.load(gamma, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                    scan7_beta = pl.load(beta, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                    scan7_normalized = pl.row_expand_mul(pl.row_expand_sub(scan7_u, mean_col), inv_std_col)
                    scan7_affine = pl.col_expand_add(pl.col_expand_mul(scan7_normalized, pl.cast(scan7_gamma, pl.FP32)), pl.cast(scan7_beta, pl.FP32))
                    scan7_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan7_u, pl.BF16)
                    pl.store(pl.set_validshape(scan7_stored_h, valid_rows, scan7_cols), [row_start, scan7_col], h)
                    scan7_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan7_normalized, pl.BF16)
                    pl.store(pl.set_validshape(scan7_stored_xhat, valid_rows, scan7_cols), [row_start, scan7_col], xhat)
                    scan7_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan7_affine, pl.BF16)
                    pl.store(pl.set_validshape(scan7_stored_y, valid_rows, scan7_cols), [row_start, scan7_col], y)
    return y


@pl.jit
def scheduled_static_bf16_1(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_group_0 in pl.spmd((rows // 32 * 32 - 0 + 31) // 32, name_hint='row_fanout_0', sync_start=False):
        row_start = 0 + row_group_0 * 32
        sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
        sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
        for scan0_col in pl.range(0, hidden // 128 * 128, 128):
            scan0_x = pl.load(x, [row_start, scan0_col], [32, 128])
            scan0_res = pl.load(res, [row_start, scan0_col], [32, 128])
            scan0_u = pl.add(pl.cast(scan0_x, pl.FP32), pl.cast(scan0_res, pl.FP32))
            sum_acc = pl.add(sum_acc, scan0_u)
            sq_acc = pl.add(sq_acc, pl.mul(scan0_u, scan0_u))
        if hidden % 128 != 0 and 0 < hidden:
            for scan1_col in pl.range(hidden // 128 * 128, hidden, 128):
                scan1_cols = pl.min(128, hidden - scan1_col)
                scan1_x = pl.load(x, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                scan1_res = pl.load(res, [row_start, scan1_col], [32, 128], valid_shape=[32, scan1_cols])
                scan1_u = pl.add(pl.cast(scan1_x, pl.FP32), pl.cast(scan1_res, pl.FP32))
                scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                scan1_u = pl.set_validshape(scan1_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan1_u)
                sq_acc = pl.add(sq_acc, pl.mul(scan1_u, scan1_u))
        square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
        mean_square_row = pl.mul(square_row, 1.0 / hidden)
        sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
        mean_row = pl.mul(sum_row, 1.0 / hidden)
        mean_col = pl.reshape(mean_row, [32, 1])
        variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
        inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
        inv_std_col = pl.reshape(inv_std_row, [32, 1])
        pl.store(pl.set_validshape(mean_row, 1, 32), [0, row_start], mu)
        pl.store(pl.set_validshape(inv_std_row, 1, 32), [0, row_start], rstd)
        for scan2_col in pl.range(0, hidden // 256 * 256, 256):
            scan2_x = pl.load(x, [row_start, scan2_col], [32, 256])
            scan2_res = pl.load(res, [row_start, scan2_col], [32, 256])
            scan2_u = pl.add(pl.cast(scan2_x, pl.FP32), pl.cast(scan2_res, pl.FP32))
            scan2_gamma = pl.load(gamma, [0, scan2_col], [1, 256])
            scan2_beta = pl.load(beta, [0, scan2_col], [1, 256])
            scan2_normalized = pl.row_expand_mul(pl.row_expand_sub(scan2_u, mean_col), inv_std_col)
            scan2_affine = pl.col_expand_add(pl.col_expand_mul(scan2_normalized, pl.cast(scan2_gamma, pl.FP32)), pl.cast(scan2_beta, pl.FP32))
            scan2_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan2_u, pl.BF16)
            pl.store(scan2_stored_h, [row_start, scan2_col], h)
            scan2_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan2_normalized, pl.BF16)
            pl.store(scan2_stored_xhat, [row_start, scan2_col], xhat)
            scan2_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan2_affine, pl.BF16)
            pl.store(scan2_stored_y, [row_start, scan2_col], y)
        if hidden % 256 != 0 and 0 < hidden:
            for scan3_col in pl.range(hidden // 256 * 256, hidden, 256):
                scan3_cols = pl.min(256, hidden - scan3_col)
                scan3_x = pl.load(x, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                scan3_res = pl.load(res, [row_start, scan3_col], [32, 256], valid_shape=[32, scan3_cols])
                scan3_u = pl.add(pl.cast(scan3_x, pl.FP32), pl.cast(scan3_res, pl.FP32))
                scan3_gamma = pl.load(gamma, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                scan3_beta = pl.load(beta, [0, scan3_col], [1, 256], valid_shape=[1, scan3_cols])
                scan3_normalized = pl.row_expand_mul(pl.row_expand_sub(scan3_u, mean_col), inv_std_col)
                scan3_affine = pl.col_expand_add(pl.col_expand_mul(scan3_normalized, pl.cast(scan3_gamma, pl.FP32)), pl.cast(scan3_beta, pl.FP32))
                scan3_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan3_u, pl.BF16)
                pl.store(pl.set_validshape(scan3_stored_h, 32, scan3_cols), [row_start, scan3_col], h)
                scan3_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan3_normalized, pl.BF16)
                pl.store(pl.set_validshape(scan3_stored_xhat, 32, scan3_cols), [row_start, scan3_col], xhat)
                scan3_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan3_affine, pl.BF16)
                pl.store(pl.set_validshape(scan3_stored_y, 32, scan3_cols), [row_start, scan3_col], y)
    if rows % 32 != 0:
        for row_group_1 in pl.spmd(1, name_hint='row_fanout_1', sync_start=False):
            row_start = rows // 32 * 32 + row_group_1 * 32
            valid_rows = pl.min(32, rows - row_start)
            sum_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            sq_acc = pl.tile.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan4_col in pl.range(0, hidden // 128 * 128, 128):
                scan4_x = pl.load(x, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                scan4_res = pl.load(res, [row_start, scan4_col], [32, 128], valid_shape=[valid_rows, 128])
                scan4_u = pl.add(pl.cast(scan4_x, pl.FP32), pl.cast(scan4_res, pl.FP32))
                scan4_u = pl.fillpad(pl.set_validshape(scan4_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                scan4_u = pl.set_validshape(scan4_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan4_u)
                sq_acc = pl.add(sq_acc, pl.mul(scan4_u, scan4_u))
            if hidden % 128 != 0 and 0 < hidden:
                for scan5_col in pl.range(hidden // 128 * 128, hidden, 128):
                    scan5_cols = pl.min(128, hidden - scan5_col)
                    scan5_x = pl.load(x, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                    scan5_res = pl.load(res, [row_start, scan5_col], [32, 128], valid_shape=[valid_rows, scan5_cols])
                    scan5_u = pl.add(pl.cast(scan5_x, pl.FP32), pl.cast(scan5_res, pl.FP32))
                    scan5_u = pl.fillpad(pl.set_validshape(scan5_u, valid_rows, scan5_cols), pad_value=pl.PadValue.zero)
                    scan5_u = pl.set_validshape(scan5_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan5_u)
                    sq_acc = pl.add(sq_acc, pl.mul(scan5_u, scan5_u))
            square_row = pl.reshape(pl.row_sum(sq_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_square_row = pl.mul(square_row, 1.0 / hidden)
            sum_row = pl.reshape(pl.row_sum(sum_acc, pl.tile.create([32, 128], pl.FP32)), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            variance_row = pl.sub(mean_square_row, pl.mul(mean_row, mean_row))
            inv_std_row = pl.tile.rsqrt(pl.add(variance_row, EPS), pl.tile.create([1, 32], pl.FP32))
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            pl.store(pl.set_validshape(mean_row, 1, valid_rows), [0, row_start], mu)
            pl.store(pl.set_validshape(inv_std_row, 1, valid_rows), [0, row_start], rstd)
            for scan6_col in pl.range(0, hidden // 256 * 256, 256):
                scan6_x = pl.load(x, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                scan6_res = pl.load(res, [row_start, scan6_col], [32, 256], valid_shape=[valid_rows, 256])
                scan6_u = pl.add(pl.cast(scan6_x, pl.FP32), pl.cast(scan6_res, pl.FP32))
                scan6_gamma = pl.load(gamma, [0, scan6_col], [1, 256])
                scan6_beta = pl.load(beta, [0, scan6_col], [1, 256])
                scan6_normalized = pl.row_expand_mul(pl.row_expand_sub(scan6_u, mean_col), inv_std_col)
                scan6_affine = pl.col_expand_add(pl.col_expand_mul(scan6_normalized, pl.cast(scan6_gamma, pl.FP32)), pl.cast(scan6_beta, pl.FP32))
                scan6_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan6_u, pl.BF16)
                pl.store(pl.set_validshape(scan6_stored_h, valid_rows, 256), [row_start, scan6_col], h)
                scan6_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan6_normalized, pl.BF16)
                pl.store(pl.set_validshape(scan6_stored_xhat, valid_rows, 256), [row_start, scan6_col], xhat)
                scan6_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan6_affine, pl.BF16)
                pl.store(pl.set_validshape(scan6_stored_y, valid_rows, 256), [row_start, scan6_col], y)
            if hidden % 256 != 0 and 0 < hidden:
                for scan7_col in pl.range(hidden // 256 * 256, hidden, 256):
                    scan7_cols = pl.min(256, hidden - scan7_col)
                    scan7_x = pl.load(x, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                    scan7_res = pl.load(res, [row_start, scan7_col], [32, 256], valid_shape=[valid_rows, scan7_cols])
                    scan7_u = pl.add(pl.cast(scan7_x, pl.FP32), pl.cast(scan7_res, pl.FP32))
                    scan7_gamma = pl.load(gamma, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                    scan7_beta = pl.load(beta, [0, scan7_col], [1, 256], valid_shape=[1, scan7_cols])
                    scan7_normalized = pl.row_expand_mul(pl.row_expand_sub(scan7_u, mean_col), inv_std_col)
                    scan7_affine = pl.col_expand_add(pl.col_expand_mul(scan7_normalized, pl.cast(scan7_gamma, pl.FP32)), pl.cast(scan7_beta, pl.FP32))
                    scan7_stored_h: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_h'), pl.Mem.Vec] = pl.cast(scan7_u, pl.BF16)
                    pl.store(pl.set_validshape(scan7_stored_h, valid_rows, scan7_cols), [row_start, scan7_col], h)
                    scan7_stored_xhat: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_xhat'), pl.Mem.Vec] = pl.cast(scan7_normalized, pl.BF16)
                    pl.store(pl.set_validshape(scan7_stored_xhat, valid_rows, scan7_cols), [row_start, scan7_col], xhat)
                    scan7_stored_y: pl.Tile[[32, 256], pl.BF16, pl.MemRef('stored_y'), pl.Mem.Vec] = pl.cast(scan7_affine, pl.BF16)
                    pl.store(pl.set_validshape(scan7_stored_y, valid_rows, scan7_cols), [row_start, scan7_col], y)
    return y


@pl.jit
def scheduled_peeled_fp32_1(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_group_0 in pl.spmd(((rows // 32 * 32) - (0) + 31) // 32, name_hint="row_fanout_0", sync_start=False):
        row_start = (0) + row_group_0 * 32
        sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
        for scan0_col in pl.range(0, (hidden) // 128 * 128, 128):
            scan0_x = pl.slice(x, [32, 128], [row_start, scan0_col], valid_shape=[32, 128])
            scan0_res = pl.slice(res, [32, 128], [row_start, scan0_col], valid_shape=[32, 128])
            scan0_u = pl.add(scan0_x, scan0_res)
            sum_acc = pl.add(sum_acc, scan0_u)
        if (hidden) % 128 != 0 and (0) < (hidden):
            for scan1_col in pl.range((hidden) // 128 * 128, hidden, 128):
                scan1_cols = pl.min(128, (hidden) - scan1_col)
                scan1_x = pl.slice(x, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                scan1_res = pl.slice(res, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                scan1_u = pl.add(scan1_x, scan1_res)
                scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                scan1_u = pl.set_validshape(scan1_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan1_u)
        sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
        mean_row = pl.mul(sum_row, 1.0 / hidden)
        mean_col = pl.reshape(mean_row, [32, 1])
        centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
        for scan2_col in pl.range(0, (hidden) // 128 * 128, 128):
            scan2_x = pl.slice(x, [32, 128], [row_start, scan2_col], valid_shape=[32, 128])
            scan2_res = pl.slice(res, [32, 128], [row_start, scan2_col], valid_shape=[32, 128])
            scan2_u = pl.add(scan2_x, scan2_res)
            scan2_centred = pl.row_expand_sub(scan2_u, mean_col)
            centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan2_centred, scan2_centred))
        if (hidden) % 128 != 0 and (0) < (hidden):
            for scan3_col in pl.range((hidden) // 128 * 128, hidden, 128):
                scan3_cols = pl.min(128, (hidden) - scan3_col)
                scan3_x = pl.slice(x, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                scan3_res = pl.slice(res, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                scan3_u = pl.add(scan3_x, scan3_res)
                scan3_centred = pl.row_expand_sub(scan3_u, mean_col)
                scan3_centred = pl.fillpad(pl.set_validshape(scan3_centred, 32, scan3_cols), pad_value=pl.PadValue.zero)
                scan3_centred = pl.set_validshape(scan3_centred, 32, 128)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan3_centred, scan3_centred))
        centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
        variance_row = pl.mul(centred_square_row, 1.0 / hidden)
        inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
        inv_std_col = pl.reshape(inv_std_row, [32, 1])
        mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, 32)
        rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, 32)
        for scan4_col in pl.range(0, (hidden) // 256 * 256, 256):
            scan4_x = pl.slice(x, [32, 256], [row_start, scan4_col], valid_shape=[32, 256])
            scan4_res = pl.slice(res, [32, 256], [row_start, scan4_col], valid_shape=[32, 256])
            scan4_u = pl.add(scan4_x, scan4_res)
            scan4_gamma = pl.slice(gamma, [1, 256], [0, scan4_col], valid_shape=[1, 256])
            scan4_beta = pl.slice(beta, [1, 256], [0, scan4_col], valid_shape=[1, 256])
            scan4_normalized = pl.row_expand_mul(pl.row_expand_sub(scan4_u, mean_col), inv_std_col)
            scan4_affine = pl.col_expand_add(pl.col_expand_mul(scan4_normalized, scan4_gamma), scan4_beta)
            h[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_u
            xhat[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_normalized
            y[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_affine
        if (hidden) % 256 != 0 and (0) < (hidden):
            for scan5_col in pl.range((hidden) // 256 * 256, hidden, 256):
                scan5_cols = pl.min(256, (hidden) - scan5_col)
                scan5_x = pl.slice(x, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                scan5_res = pl.slice(res, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                scan5_u = pl.add(scan5_x, scan5_res)
                scan5_gamma = pl.slice(gamma, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                scan5_beta = pl.slice(beta, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                scan5_normalized = pl.row_expand_mul(pl.row_expand_sub(scan5_u, mean_col), inv_std_col)
                scan5_affine = pl.col_expand_add(pl.col_expand_mul(scan5_normalized, scan5_gamma), scan5_beta)
                h[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_u, 32, scan5_cols)
                xhat[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_normalized, 32, scan5_cols)
                y[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_affine, 32, scan5_cols)
    if rows % 32 != 0:
        for row_group_1 in pl.spmd(1, name_hint="row_fanout_1", sync_start=False):
            row_start = (rows // 32 * 32) + row_group_1 * 32
            valid_rows = pl.min(32, rows - row_start)
            sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan6_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan6_x = pl.slice(x, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                scan6_res = pl.slice(res, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                scan6_u = pl.add(scan6_x, scan6_res)
                scan6_u = pl.fillpad(pl.set_validshape(scan6_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                scan6_u = pl.set_validshape(scan6_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan6_u)
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan7_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan7_cols = pl.min(128, (hidden) - scan7_col)
                    scan7_x = pl.slice(x, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                    scan7_res = pl.slice(res, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                    scan7_u = pl.add(scan7_x, scan7_res)
                    scan7_u = pl.fillpad(pl.set_validshape(scan7_u, valid_rows, scan7_cols), pad_value=pl.PadValue.zero)
                    scan7_u = pl.set_validshape(scan7_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan7_u)
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan8_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan8_x = pl.slice(x, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                scan8_res = pl.slice(res, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                scan8_u = pl.add(scan8_x, scan8_res)
                scan8_centred = pl.row_expand_sub(scan8_u, mean_col)
                scan8_centred = pl.fillpad(pl.set_validshape(scan8_centred, valid_rows, 128), pad_value=pl.PadValue.zero)
                scan8_centred = pl.set_validshape(scan8_centred, 32, 128)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan8_centred, scan8_centred))
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan9_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan9_cols = pl.min(128, (hidden) - scan9_col)
                    scan9_x = pl.slice(x, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                    scan9_res = pl.slice(res, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                    scan9_u = pl.add(scan9_x, scan9_res)
                    scan9_centred = pl.row_expand_sub(scan9_u, mean_col)
                    scan9_centred = pl.fillpad(pl.set_validshape(scan9_centred, valid_rows, scan9_cols), pad_value=pl.PadValue.zero)
                    scan9_centred = pl.set_validshape(scan9_centred, 32, 128)
                    centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan9_centred, scan9_centred))
            centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
            variance_row = pl.mul(centred_square_row, 1.0 / hidden)
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, valid_rows)
            for scan10_col in pl.range(0, (hidden) // 256 * 256, 256):
                scan10_x = pl.slice(x, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                scan10_res = pl.slice(res, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                scan10_u = pl.add(scan10_x, scan10_res)
                scan10_gamma = pl.slice(gamma, [1, 256], [0, scan10_col], valid_shape=[1, 256])
                scan10_beta = pl.slice(beta, [1, 256], [0, scan10_col], valid_shape=[1, 256])
                scan10_normalized = pl.row_expand_mul(pl.row_expand_sub(scan10_u, mean_col), inv_std_col)
                scan10_affine = pl.col_expand_add(pl.col_expand_mul(scan10_normalized, scan10_gamma), scan10_beta)
                h[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_u, valid_rows, 256)
                xhat[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_normalized, valid_rows, 256)
                y[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_affine, valid_rows, 256)
            if (hidden) % 256 != 0 and (0) < (hidden):
                for scan11_col in pl.range((hidden) // 256 * 256, hidden, 256):
                    scan11_cols = pl.min(256, (hidden) - scan11_col)
                    scan11_x = pl.slice(x, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                    scan11_res = pl.slice(res, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                    scan11_u = pl.add(scan11_x, scan11_res)
                    scan11_gamma = pl.slice(gamma, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                    scan11_beta = pl.slice(beta, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                    scan11_normalized = pl.row_expand_mul(pl.row_expand_sub(scan11_u, mean_col), inv_std_col)
                    scan11_affine = pl.col_expand_add(pl.col_expand_mul(scan11_normalized, scan11_gamma), scan11_beta)
                    h[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_u, valid_rows, scan11_cols)
                    xhat[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_normalized, valid_rows, scan11_cols)
                    y[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_affine, valid_rows, scan11_cols)
    return y


@pl.jit
def scheduled_static_fp32_1(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, h: pl.Out[pl.Tensor], xhat: pl.Out[pl.Tensor], mu: pl.Out[pl.Tensor], rstd: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_group_0 in pl.spmd(((rows // 32 * 32) - (0) + 31) // 32, name_hint="row_fanout_0", sync_start=False):
        row_start = (0) + row_group_0 * 32
        sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
        for scan0_col in pl.range(0, (hidden) // 128 * 128, 128):
            scan0_x = pl.slice(x, [32, 128], [row_start, scan0_col])
            scan0_res = pl.slice(res, [32, 128], [row_start, scan0_col])
            scan0_u = pl.add(scan0_x, scan0_res)
            sum_acc = pl.add(sum_acc, scan0_u)
        if (hidden) % 128 != 0 and (0) < (hidden):
            for scan1_col in pl.range((hidden) // 128 * 128, hidden, 128):
                scan1_cols = pl.min(128, (hidden) - scan1_col)
                scan1_x = pl.slice(x, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                scan1_res = pl.slice(res, [32, 128], [row_start, scan1_col], valid_shape=[32, scan1_cols])
                scan1_u = pl.add(scan1_x, scan1_res)
                scan1_u = pl.fillpad(pl.set_validshape(scan1_u, 32, scan1_cols), pad_value=pl.PadValue.zero)
                scan1_u = pl.set_validshape(scan1_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan1_u)
        sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
        mean_row = pl.mul(sum_row, 1.0 / hidden)
        mean_col = pl.reshape(mean_row, [32, 1])
        centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
        for scan2_col in pl.range(0, (hidden) // 128 * 128, 128):
            scan2_x = pl.slice(x, [32, 128], [row_start, scan2_col])
            scan2_res = pl.slice(res, [32, 128], [row_start, scan2_col])
            scan2_u = pl.add(scan2_x, scan2_res)
            scan2_centred = pl.row_expand_sub(scan2_u, mean_col)
            centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan2_centred, scan2_centred))
        if (hidden) % 128 != 0 and (0) < (hidden):
            for scan3_col in pl.range((hidden) // 128 * 128, hidden, 128):
                scan3_cols = pl.min(128, (hidden) - scan3_col)
                scan3_x = pl.slice(x, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                scan3_res = pl.slice(res, [32, 128], [row_start, scan3_col], valid_shape=[32, scan3_cols])
                scan3_u = pl.add(scan3_x, scan3_res)
                scan3_centred = pl.row_expand_sub(scan3_u, mean_col)
                scan3_centred = pl.fillpad(pl.set_validshape(scan3_centred, 32, scan3_cols), pad_value=pl.PadValue.zero)
                scan3_centred = pl.set_validshape(scan3_centred, 32, 128)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan3_centred, scan3_centred))
        centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
        variance_row = pl.mul(centred_square_row, 1.0 / hidden)
        inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
        inv_std_col = pl.reshape(inv_std_row, [32, 1])
        mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, 32)
        rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, 32)
        for scan4_col in pl.range(0, (hidden) // 256 * 256, 256):
            scan4_x = pl.slice(x, [32, 256], [row_start, scan4_col])
            scan4_res = pl.slice(res, [32, 256], [row_start, scan4_col])
            scan4_u = pl.add(scan4_x, scan4_res)
            scan4_gamma = pl.slice(gamma, [1, 256], [0, scan4_col])
            scan4_beta = pl.slice(beta, [1, 256], [0, scan4_col])
            scan4_normalized = pl.row_expand_mul(pl.row_expand_sub(scan4_u, mean_col), inv_std_col)
            scan4_affine = pl.col_expand_add(pl.col_expand_mul(scan4_normalized, scan4_gamma), scan4_beta)
            h[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_u
            xhat[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_normalized
            y[row_start:row_start + 32, scan4_col:scan4_col + 256] = scan4_affine
        if (hidden) % 256 != 0 and (0) < (hidden):
            for scan5_col in pl.range((hidden) // 256 * 256, hidden, 256):
                scan5_cols = pl.min(256, (hidden) - scan5_col)
                scan5_x = pl.slice(x, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                scan5_res = pl.slice(res, [32, 256], [row_start, scan5_col], valid_shape=[32, scan5_cols])
                scan5_u = pl.add(scan5_x, scan5_res)
                scan5_gamma = pl.slice(gamma, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                scan5_beta = pl.slice(beta, [1, 256], [0, scan5_col], valid_shape=[1, scan5_cols])
                scan5_normalized = pl.row_expand_mul(pl.row_expand_sub(scan5_u, mean_col), inv_std_col)
                scan5_affine = pl.col_expand_add(pl.col_expand_mul(scan5_normalized, scan5_gamma), scan5_beta)
                h[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_u, 32, scan5_cols)
                xhat[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_normalized, 32, scan5_cols)
                y[row_start:row_start + 32, scan5_col:scan5_col + 256] = pl.set_validshape(scan5_affine, 32, scan5_cols)
    if rows % 32 != 0:
        for row_group_1 in pl.spmd(1, name_hint="row_fanout_1", sync_start=False):
            row_start = (rows // 32 * 32) + row_group_1 * 32
            valid_rows = pl.min(32, rows - row_start)
            sum_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan6_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan6_x = pl.slice(x, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                scan6_res = pl.slice(res, [32, 128], [row_start, scan6_col], valid_shape=[valid_rows, 128])
                scan6_u = pl.add(scan6_x, scan6_res)
                scan6_u = pl.fillpad(pl.set_validshape(scan6_u, valid_rows, 128), pad_value=pl.PadValue.zero)
                scan6_u = pl.set_validshape(scan6_u, 32, 128)
                sum_acc = pl.add(sum_acc, scan6_u)
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan7_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan7_cols = pl.min(128, (hidden) - scan7_col)
                    scan7_x = pl.slice(x, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                    scan7_res = pl.slice(res, [32, 128], [row_start, scan7_col], valid_shape=[valid_rows, scan7_cols])
                    scan7_u = pl.add(scan7_x, scan7_res)
                    scan7_u = pl.fillpad(pl.set_validshape(scan7_u, valid_rows, scan7_cols), pad_value=pl.PadValue.zero)
                    scan7_u = pl.set_validshape(scan7_u, 32, 128)
                    sum_acc = pl.add(sum_acc, scan7_u)
            sum_row = pl.reshape(pl.row_sum(sum_acc), [1, 32])
            mean_row = pl.mul(sum_row, 1.0 / hidden)
            mean_col = pl.reshape(mean_row, [32, 1])
            centred_sq_acc = pl.full([32, 128], dtype=pl.FP32, value=0.0)
            for scan8_col in pl.range(0, (hidden) // 128 * 128, 128):
                scan8_x = pl.slice(x, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                scan8_res = pl.slice(res, [32, 128], [row_start, scan8_col], valid_shape=[valid_rows, 128])
                scan8_u = pl.add(scan8_x, scan8_res)
                scan8_centred = pl.row_expand_sub(scan8_u, mean_col)
                scan8_centred = pl.fillpad(pl.set_validshape(scan8_centred, valid_rows, 128), pad_value=pl.PadValue.zero)
                scan8_centred = pl.set_validshape(scan8_centred, 32, 128)
                centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan8_centred, scan8_centred))
            if (hidden) % 128 != 0 and (0) < (hidden):
                for scan9_col in pl.range((hidden) // 128 * 128, hidden, 128):
                    scan9_cols = pl.min(128, (hidden) - scan9_col)
                    scan9_x = pl.slice(x, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                    scan9_res = pl.slice(res, [32, 128], [row_start, scan9_col], valid_shape=[valid_rows, scan9_cols])
                    scan9_u = pl.add(scan9_x, scan9_res)
                    scan9_centred = pl.row_expand_sub(scan9_u, mean_col)
                    scan9_centred = pl.fillpad(pl.set_validshape(scan9_centred, valid_rows, scan9_cols), pad_value=pl.PadValue.zero)
                    scan9_centred = pl.set_validshape(scan9_centred, 32, 128)
                    centred_sq_acc = pl.add(centred_sq_acc, pl.mul(scan9_centred, scan9_centred))
            centred_square_row = pl.reshape(pl.row_sum(centred_sq_acc), [1, 32])
            variance_row = pl.mul(centred_square_row, 1.0 / hidden)
            inv_std_row = pl.rsqrt(pl.add(variance_row, EPS), high_precision=True)
            inv_std_col = pl.reshape(inv_std_row, [32, 1])
            mu[0:1, row_start:row_start + 32] = pl.set_validshape(mean_row, 1, valid_rows)
            rstd[0:1, row_start:row_start + 32] = pl.set_validshape(inv_std_row, 1, valid_rows)
            for scan10_col in pl.range(0, (hidden) // 256 * 256, 256):
                scan10_x = pl.slice(x, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                scan10_res = pl.slice(res, [32, 256], [row_start, scan10_col], valid_shape=[valid_rows, 256])
                scan10_u = pl.add(scan10_x, scan10_res)
                scan10_gamma = pl.slice(gamma, [1, 256], [0, scan10_col])
                scan10_beta = pl.slice(beta, [1, 256], [0, scan10_col])
                scan10_normalized = pl.row_expand_mul(pl.row_expand_sub(scan10_u, mean_col), inv_std_col)
                scan10_affine = pl.col_expand_add(pl.col_expand_mul(scan10_normalized, scan10_gamma), scan10_beta)
                h[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_u, valid_rows, 256)
                xhat[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_normalized, valid_rows, 256)
                y[row_start:row_start + 32, scan10_col:scan10_col + 256] = pl.set_validshape(scan10_affine, valid_rows, 256)
            if (hidden) % 256 != 0 and (0) < (hidden):
                for scan11_col in pl.range((hidden) // 256 * 256, hidden, 256):
                    scan11_cols = pl.min(256, (hidden) - scan11_col)
                    scan11_x = pl.slice(x, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                    scan11_res = pl.slice(res, [32, 256], [row_start, scan11_col], valid_shape=[valid_rows, scan11_cols])
                    scan11_u = pl.add(scan11_x, scan11_res)
                    scan11_gamma = pl.slice(gamma, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                    scan11_beta = pl.slice(beta, [1, 256], [0, scan11_col], valid_shape=[1, scan11_cols])
                    scan11_normalized = pl.row_expand_mul(pl.row_expand_sub(scan11_u, mean_col), inv_std_col)
                    scan11_affine = pl.col_expand_add(pl.col_expand_mul(scan11_normalized, scan11_gamma), scan11_beta)
                    h[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_u, valid_rows, scan11_cols)
                    xhat[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_normalized, valid_rows, scan11_cols)
                    y[row_start:row_start + 32, scan11_col:scan11_col + 256] = pl.set_validshape(scan11_affine, valid_rows, scan11_cols)
    return y

def build(shape):
    import torch
    from golden import TensorSpec
    rows, hidden = (int(shape['M']), int(shape['N']))
    dtype_name = shape.get('dtype', 'fp32')
    if rows <= 0 or hidden <= 0:
        raise ValueError('M and N must be positive')
    if dtype_name not in ('fp32', 'bf16'):
        raise ValueError('dtype must be fp32 or bf16')
    storage_dtype = torch.bfloat16 if dtype_name == 'bf16' else torch.float32
    specs = [TensorSpec('x', [rows, hidden], storage_dtype, init_value=torch.randn), TensorSpec('res', [rows, hidden], storage_dtype, init_value=torch.randn), TensorSpec('gamma', [1, hidden], storage_dtype, init_value=torch.randn), TensorSpec('beta', [1, hidden], storage_dtype, init_value=torch.randn), TensorSpec('h', [rows, hidden], storage_dtype), TensorSpec('xhat', [rows, hidden], storage_dtype), TensorSpec('mu', [1, rows], torch.float32), TensorSpec('rstd', [1, rows], torch.float32), TensorSpec('y', [rows, hidden], storage_dtype)]

    def golden_fn(tensors):
        residual = tensors['x'].float() + tensors['res'].float()
        mean = residual.mean(dim=-1, keepdim=True)
        variance = residual.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(variance + EPS)
        normalized = (residual - mean) * inv_std
        affine = normalized * tensors['gamma'].float() + tensors['beta'].float()
        tensors['h'][:] = residual.to(storage_dtype)
        tensors['xhat'][:] = normalized.to(storage_dtype)
        tensors['mu'][:] = mean.reshape(1, -1)
        tensors['rstd'][:] = inv_std.reshape(1, -1)
        tensors['y'][:] = affine.to(storage_dtype)
    aligned = rows % ROW_TILE == 0 and hidden % STATS_COL_TILE == 0 and hidden % APPLY_COL_TILE == 0
    if aligned:
        kernel = skip_layernorm_bf16_aligned if dtype_name == 'bf16' else skip_layernorm_fp32_aligned
    else:
        kernel = skip_layernorm_bf16 if dtype_name == 'bf16' else skip_layernorm_fp32
    if rows >= 32 and hidden >= 256:
        kernel = peeled_bf16 if dtype_name == 'bf16' else peeled_fp32
    if hidden == 1:
        kernel = singleton_bf16 if dtype_name == 'bf16' else singleton_fp32
    if rows >= 32 and hidden >= 256 and not aligned:
        kernel = static_bf16 if dtype_name == 'bf16' else static_fp32
    if hidden == 1:
        kernel = vector_singleton_bf16 if dtype_name == 'bf16' else vector_singleton_fp32
    if rows >= 32 and hidden >= 256:
        row_tiles = (rows + 31) // 32
        effective_tiles = 1
        if effective_tiles > 1 and (row_tiles + effective_tiles - 1) // effective_tiles < 48:
            effective_tiles = 1
        if aligned:
            kernel = scheduled_peeled_bf16_1 if dtype_name == 'bf16' else scheduled_peeled_fp32_1
        else:
            kernel = scheduled_static_bf16_1 if dtype_name == 'bf16' else scheduled_static_fp32_1
    tolerance = 0.02 if dtype_name == 'bf16' else 0.0001
    return (kernel, specs, golden_fn, tolerance, tolerance)

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
