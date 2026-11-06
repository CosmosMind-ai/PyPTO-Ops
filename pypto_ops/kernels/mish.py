# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contiguous fused Mish with requested storage dtype and FP32 arithmetic."""

import pypto.language as pl


def golden_mish(tensors):
    import torch

    reference_x = tensors["x"].float()
    tensors["y"][:] = (
        reference_x * torch.tanh(torch.nn.functional.softplus(reference_x))
    ).to(tensors["y"].dtype)


def build(shape):
    import torch
    from golden import TensorSpec

    rows = int(shape["M"])
    cols = int(shape["N"])
    dtype_name = shape.get("dtype", "fp32")
    if rows <= 0 or cols <= 0 or dtype_name not in ("bf16", "fp32"):
        raise ValueError("Mish requires positive dimensions and bf16 or fp32")
    storage_type = pl.BF16 if dtype_name == "bf16" else pl.FP32
    torch_type = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    elements = rows * cols
    group_rows = 64
    group_span = group_rows * cols
    group_count = (rows + group_rows - 1) // group_rows
    chunk_elements = min(8192, max(16, 2 ** (elements.bit_length() - 1)))
    partition_unit = 2 * chunk_elements
    partition_units = (elements + partition_unit - 1) // partition_unit
    if dtype_name == "bf16" and elements >= partition_unit:
        group_count = min(48, partition_units)
    tolerance = 1e-2 if dtype_name == "bf16" else 1e-4

    if elements < 16 and dtype_name == "bf16":
        @pl.program
        class MishScalarBF16Program:
            @pl.function
            def mish(
                self,
                x: pl.Tensor[[rows, cols], storage_type],
                y: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            ):
                flat_x = pl.reshape(x, [1, elements])
                flat_y = pl.reshape(y, [1, elements])
                for group in pl.spmd(1, name_hint="mish_scalar"):
                    for offset in pl.range(0, elements, 1):
                        scalar_x = pl.read(flat_x, [0, offset])
                        scalar_tile = pl.write(pl.full([1, 16], dtype=storage_type, value=0.0), [0, 0], scalar_x)
                        tile_x_fp32 = pl.cast(scalar_tile, target_type=pl.FP32)
                        bounded_x = pl.minimum(tile_x_fp32, 20.0)
                        exp_x = pl.exp(bounded_x)
                        numerator = pl.mul(exp_x, pl.add(exp_x, 2.0))
                        denominator = pl.add(numerator, 2.0)
                        activation = pl.div(numerator, denominator)
                        result_fp32 = pl.mul(tile_x_fp32, activation)
                        stored_tile = pl.cast(result_fp32, target_type=pl.BF16)
                        scalar_result = pl.read(stored_tile, [0, 0])
                        flat_y = pl.write(flat_y, [0, offset], scalar_result)
                return y

        program = MishScalarBF16Program
    elif elements < 16:
        @pl.program
        class MishScalarFP32Program:
            @pl.function
            def mish(
                self,
                x: pl.Tensor[[rows, cols], storage_type],
                y: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            ):
                flat_x = pl.reshape(x, [1, elements])
                flat_y = pl.reshape(y, [1, elements])
                for group in pl.spmd(1, name_hint="mish_scalar"):
                    for offset in pl.range(0, elements, 1):
                        scalar_x = pl.read(flat_x, [0, offset])
                        scalar_tile = pl.write(pl.full([1, 16], dtype=storage_type, value=0.0), [0, 0], scalar_x)
                        tile_x_fp32 = scalar_tile
                        bounded_x = pl.minimum(tile_x_fp32, 20.0)
                        exp_x = pl.exp(bounded_x)
                        numerator = pl.mul(exp_x, pl.add(exp_x, 2.0))
                        denominator = pl.add(numerator, 2.0)
                        activation = pl.div(numerator, denominator)
                        result_fp32 = pl.mul(tile_x_fp32, activation)
                        stored_tile = result_fp32
                        scalar_result = pl.read(stored_tile, [0, 0])
                        flat_y = pl.write(flat_y, [0, offset], scalar_result)
                return y

        program = MishScalarFP32Program
    elif dtype_name == "bf16" and elements >= 2 * chunk_elements:
        @pl.program
        class MishTransferBF16Program:
            @pl.function
            def mish(
                self,
                x: pl.Tensor[[rows, cols], storage_type],
                y: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            ):
                flat_x = pl.reshape(x, [1, elements])
                flat_y = pl.reshape(y, [1, elements])
                for group in pl.spmd(group_count, name_hint="mish_contiguous"):
                    group_base = ((group * partition_units) // group_count) * partition_unit
                    group_end = pl.min((((group + 1) * partition_units) // group_count) * partition_unit, elements)
                    batch_span = 2 * chunk_elements
                    tail_start = group_base + ((group_end - group_base) // batch_span) * batch_span
                    for batch_base in pl.range(group_base, tail_start, batch_span):
                        offset_0 = batch_base + 0 * chunk_elements
                        input_0: pl.Tile[[1, chunk_elements], pl.BF16, pl.MemRef("input_0"), pl.Mem.Vec] = pl.load(flat_x, [0, offset_0], [1, chunk_elements])
                        offset_1 = batch_base + 1 * chunk_elements
                        input_1: pl.Tile[[1, chunk_elements], pl.BF16, pl.MemRef("input_1"), pl.Mem.Vec] = pl.load(flat_x, [0, offset_1], [1, chunk_elements])
                        x_fp32_0 = pl.cast(input_0, target_type=pl.FP32)
                        bounded_0 = pl.minimum(x_fp32_0, 20.0)
                        exp_0 = pl.exp(bounded_0)
                        numerator_0 = pl.mul(exp_0, pl.add(exp_0, 2.0))
                        denominator_0 = pl.add(numerator_0, 2.0)
                        activation_0 = pl.div(numerator_0, denominator_0)
                        result_fp32_0 = pl.mul(x_fp32_0, activation_0)
                        result_0: pl.Tile[[1, chunk_elements], pl.BF16, pl.MemRef("output_0"), pl.Mem.Vec] = pl.cast(result_fp32_0, target_type=pl.BF16)
                        flat_y = pl.store(result_0, [0, offset_0], flat_y)
                        x_fp32_1 = pl.cast(input_1, target_type=pl.FP32)
                        bounded_1 = pl.minimum(x_fp32_1, 20.0)
                        exp_1 = pl.exp(bounded_1)
                        numerator_1 = pl.mul(exp_1, pl.add(exp_1, 2.0))
                        denominator_1 = pl.add(numerator_1, 2.0)
                        activation_1 = pl.div(numerator_1, denominator_1)
                        result_fp32_1 = pl.mul(x_fp32_1, activation_1)
                        result_1: pl.Tile[[1, chunk_elements], pl.BF16, pl.MemRef("output_1"), pl.Mem.Vec] = pl.cast(result_fp32_1, target_type=pl.BF16)
                        flat_y = pl.store(result_1, [0, offset_1], flat_y)
                    for offset in pl.range(tail_start, group_end, chunk_elements):
                        tile_x = pl.slice(flat_x, [1, chunk_elements], [0, offset], valid_shape=[1, pl.min(chunk_elements, group_end - offset)])
                        x_fp32_tail = pl.cast(tile_x, target_type=pl.FP32)
                        bounded_tail = pl.minimum(x_fp32_tail, 20.0)
                        exp_tail = pl.exp(bounded_tail)
                        numerator_tail = pl.mul(exp_tail, pl.add(exp_tail, 2.0))
                        denominator_tail = pl.add(numerator_tail, 2.0)
                        activation_tail = pl.div(numerator_tail, denominator_tail)
                        result_fp32_tail = pl.mul(x_fp32_tail, activation_tail)
                        result_tail = pl.cast(result_fp32_tail, target_type=pl.BF16)
                        flat_y = pl.assemble(flat_y, result_tail, [0, offset])
                return y

        program = MishTransferBF16Program
    elif dtype_name == "bf16":
        @pl.program
        class MishTransferBF16SerialProgram:
            @pl.function
            def mish(
                self,
                x: pl.Tensor[[rows, cols], storage_type],
                y: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            ):
                flat_x = pl.reshape(x, [1, elements])
                flat_y = pl.reshape(y, [1, elements])
                for group in pl.spmd(group_count, name_hint="mish_contiguous"):
                    group_base = group * group_span
                    group_end = pl.min(group_base + group_span, elements)
                    tail_start = group_base
                    for offset in pl.range(tail_start, group_end, chunk_elements):
                        tile_x = pl.slice(flat_x, [1, chunk_elements], [0, offset], valid_shape=[1, pl.min(chunk_elements, group_end - offset)])
                        x_fp32_tail = pl.cast(tile_x, target_type=pl.FP32)
                        bounded_tail = pl.minimum(x_fp32_tail, 20.0)
                        exp_tail = pl.exp(bounded_tail)
                        numerator_tail = pl.mul(exp_tail, pl.add(exp_tail, 2.0))
                        denominator_tail = pl.add(numerator_tail, 2.0)
                        activation_tail = pl.div(numerator_tail, denominator_tail)
                        result_fp32_tail = pl.mul(x_fp32_tail, activation_tail)
                        result_tail = pl.cast(result_fp32_tail, target_type=pl.BF16)
                        flat_y = pl.assemble(flat_y, result_tail, [0, offset])
                return y

        program = MishTransferBF16SerialProgram
    else:
        @pl.program
        class MishTransferFP32Program:
            @pl.function
            def mish(
                self,
                x: pl.Tensor[[rows, cols], storage_type],
                y: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            ):
                flat_x = pl.reshape(x, [1, elements])
                flat_y = pl.reshape(y, [1, elements])
                for group in pl.spmd(group_count, name_hint="mish_contiguous"):
                    group_base = group * group_span
                    group_end = pl.min(group_base + group_span, elements)
                    tail_start = group_base
                    for offset in pl.range(tail_start, group_end, chunk_elements):
                        tile_x = pl.slice(flat_x, [1, chunk_elements], [0, offset], valid_shape=[1, pl.min(chunk_elements, group_end - offset)])
                        x_fp32_tail = tile_x
                        bounded_tail = pl.minimum(x_fp32_tail, 20.0)
                        exp_tail = pl.exp(bounded_tail)
                        numerator_tail = pl.mul(exp_tail, pl.add(exp_tail, 2.0))
                        denominator_tail = pl.add(numerator_tail, 2.0)
                        activation_tail = pl.div(numerator_tail, denominator_tail)
                        result_fp32_tail = pl.mul(x_fp32_tail, activation_tail)
                        result_tail = result_fp32_tail
                        flat_y = pl.assemble(flat_y, result_tail, [0, offset])
                return y

        program = MishTransferFP32Program

    specs = [
        TensorSpec("x", [rows, cols], torch_type, init_value=torch.randn),
        TensorSpec("y", [rows, cols], torch_type),
    ]
    return program, specs, golden_mish, tolerance, tolerance

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
