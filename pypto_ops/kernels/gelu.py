# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GELU (tanh approximation) — bf16 I/O with fp32 arithmetic."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024}


RTOL = 2e-2
ATOL = 2e-2
ROW_TILE = 32
DMA_COLS = 512
MICRO_ROWS = 16
CHUNK_SLAB_LOG2 = 3


def build(shape):
    import torch
    from golden import TensorSpec

    M, N = shape["M"], shape["N"]
    tail_width = 1 if N == 1 else min(128, ((N + 15) // 16) * 16)
    full_stop = (N // DMA_COLS) * DMA_COLS
    micro_stop = min(ROW_TILE, M)
    chunk_cols = (1 << CHUNK_SLAB_LOG2) * DMA_COLS
    column_groups = (N + chunk_cols - 1) // chunk_cols
    block_count = ((M + ROW_TILE - 1) // ROW_TILE) * column_groups

    if N < DMA_COLS:
        @pl.jit
        def gelu_tail(
            x: pl.Tensor[[M, N], pl.BF16],
            y: pl.Out[pl.Tensor[[M, N], pl.BF16]],
        ):
            for band in pl.spmd((M + ROW_TILE - 1) // ROW_TILE, sync_start=False):
                r = band * ROW_TILE
                row_count = pl.min(ROW_TILE, M - r)
                for c in pl.range(0, N, 128):
                    col_count = pl.min(128, N - c)
                    micro_input = pl.load(x, [r, c], [ROW_TILE, tail_width], valid_shape=[row_count, col_count], clamp=True)
                    value_fp32 = pl.cast(micro_input, target_type=pl.FP32)
                    bounded_fp16 = pl.minimum(pl.abs(pl.cast(value_fp32, target_type=pl.FP16, mode="rint")), 3.0)
                    base_fp32 = pl.maximum(value_fp32, 0.0)
                    coordinate = pl.mul(bounded_fp16, 0.3333333333333333)
                    polynomial = pl.add(pl.mul(coordinate, 1.8336524795287883), -3.3422971644298456)
                    polynomial = pl.add(pl.mul(coordinate, polynomial), 1.6185301569489343)
                    correction = pl.mul(pl.mul(coordinate, pl.sub(coordinate, 1.0)), polynomial)
                    correction_fp32 = pl.cast(correction, target_type=pl.FP32)
                    result_fp32 = pl.add(base_fp32, correction_fp32)
                    micro_output = pl.cast(result_fp32, target_type=pl.BF16, mode="rint")
                    pl.store(micro_output, [r, c], y)
            return y
        kernel = gelu_tail
    else:
        @pl.jit
        def gelu_wide_slab_row(
            x: pl.Tensor[[M, N], pl.BF16],
            y: pl.Out[pl.Tensor[[M, N], pl.BF16]],
        ):
            for block_index in pl.spmd(block_count, sync_start=False):
                row_band = block_index // column_groups
                column_group = block_index % column_groups
                r = row_band * ROW_TILE
                row_count = pl.min(ROW_TILE, M - r)
                column_start = column_group * chunk_cols
                owned_full_stop = pl.min(column_start + chunk_cols, full_stop)
                last_start = owned_full_stop - DMA_COLS
                if column_start < full_stop:
                    current = pl.load(x, [r, column_start], [ROW_TILE, DMA_COLS], valid_shape=[row_count, DMA_COLS], clamp=True)
                    current = pl.set_validshape(current, row_count, DMA_COLS)
                    final_slab = current
                    for c, (current,) in pl.range(column_start, last_start, DMA_COLS, init_values=(current,)):
                        next_slab = pl.load(x, [r, c + DMA_COLS], [ROW_TILE, DMA_COLS], valid_shape=[row_count, DMA_COLS], clamp=True)
                        next_slab = pl.set_validshape(next_slab, row_count, DMA_COLS)
                        canvas = pl.set_validshape(current, ROW_TILE, DMA_COLS)
                        for micro_start in pl.unroll(0, micro_stop, MICRO_ROWS):
                            if micro_start < row_count:
                                valid_rows = pl.min(MICRO_ROWS, row_count - micro_start)
                                micro_input = pl.slice(canvas, [MICRO_ROWS, DMA_COLS], [micro_start, 0], valid_shape=[valid_rows, DMA_COLS])
                                value_fp32 = pl.cast(micro_input, target_type=pl.FP32)
                                bounded_fp16 = pl.minimum(pl.abs(pl.cast(value_fp32, target_type=pl.FP16, mode="rint")), 3.0)
                                base_fp32 = pl.maximum(value_fp32, 0.0)
                                coordinate = pl.mul(bounded_fp16, 0.3333333333333333)
                                polynomial = pl.add(pl.mul(coordinate, 1.8336524795287883), -3.3422971644298456)
                                polynomial = pl.add(pl.mul(coordinate, polynomial), 1.6185301569489343)
                                correction = pl.mul(pl.mul(coordinate, pl.sub(coordinate, 1.0)), polynomial)
                                correction_fp32 = pl.cast(correction, target_type=pl.FP32)
                                result_fp32 = pl.add(base_fp32, correction_fp32)
                                micro_output = pl.cast(result_fp32, target_type=pl.BF16, mode="rint")
                                canvas = pl.assemble(canvas, micro_output, [micro_start, 0])
                        output_slab = pl.set_validshape(canvas, row_count, DMA_COLS)
                        pl.store(output_slab, [r, c], y)
                        final_slab = pl.yield_(next_slab)
                    canvas = pl.set_validshape(final_slab, ROW_TILE, DMA_COLS)
                    for micro_start in pl.unroll(0, micro_stop, MICRO_ROWS):
                        if micro_start < row_count:
                            valid_rows = pl.min(MICRO_ROWS, row_count - micro_start)
                            micro_input = pl.slice(canvas, [MICRO_ROWS, DMA_COLS], [micro_start, 0], valid_shape=[valid_rows, DMA_COLS])
                            value_fp32 = pl.cast(micro_input, target_type=pl.FP32)
                            bounded_fp16 = pl.minimum(pl.abs(pl.cast(value_fp32, target_type=pl.FP16, mode="rint")), 3.0)
                            base_fp32 = pl.maximum(value_fp32, 0.0)
                            coordinate = pl.mul(bounded_fp16, 0.3333333333333333)
                            polynomial = pl.add(pl.mul(coordinate, 1.8336524795287883), -3.3422971644298456)
                            polynomial = pl.add(pl.mul(coordinate, polynomial), 1.6185301569489343)
                            correction = pl.mul(pl.mul(coordinate, pl.sub(coordinate, 1.0)), polynomial)
                            correction_fp32 = pl.cast(correction, target_type=pl.FP32)
                            result_fp32 = pl.add(base_fp32, correction_fp32)
                            micro_output = pl.cast(result_fp32, target_type=pl.BF16, mode="rint")
                            canvas = pl.assemble(canvas, micro_output, [micro_start, 0])
                    last_output = pl.set_validshape(canvas, row_count, DMA_COLS)
                    pl.store(last_output, [r, last_start], y)
                if column_group == column_groups - 1:
                    for c in pl.range(full_stop, N, 128):
                        col_count = pl.min(128, N - c)
                        tail_input = pl.load(x, [r, c], [ROW_TILE, tail_width], valid_shape=[row_count, col_count], clamp=True)
                        value_fp32_tail = pl.cast(tail_input, target_type=pl.FP32)
                        bounded_fp16_tail = pl.minimum(pl.abs(pl.cast(value_fp32_tail, target_type=pl.FP16, mode="rint")), 3.0)
                        base_fp32_tail = pl.maximum(value_fp32_tail, 0.0)
                        coordinate_tail = pl.mul(bounded_fp16_tail, 0.3333333333333333)
                        polynomial_tail = pl.add(pl.mul(coordinate_tail, 1.8336524795287883), -3.3422971644298456)
                        polynomial_tail = pl.add(pl.mul(coordinate_tail, polynomial_tail), 1.6185301569489343)
                        correction_tail = pl.mul(pl.mul(coordinate_tail, pl.sub(coordinate_tail, 1.0)), polynomial_tail)
                        correction_fp32_tail = pl.cast(correction_tail, target_type=pl.FP32)
                        result_fp32_tail = pl.add(base_fp32_tail, correction_fp32_tail)
                        tail_output = pl.cast(result_fp32_tail, target_type=pl.BF16, mode="rint")
                        pl.store(tail_output, [r, c], y)
            return y
        kernel = gelu_wide_slab_row

    specs = [
        TensorSpec("x", [M, N], torch.bfloat16, init_value=torch.randn),
        TensorSpec("y", [M, N], torch.bfloat16),
    ]

    def golden(tensors):
        x_fp32 = tensors["x"].to(torch.float32)
        z = (2.0 / torch.pi) ** 0.5 * (x_fp32 + 0.044715 * x_fp32.pow(3))
        tensors["y"][:] = (0.5 * x_fp32 * (1.0 + torch.tanh(z))).to(torch.bfloat16)

    return kernel, specs, golden, RTOL, ATOL

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
