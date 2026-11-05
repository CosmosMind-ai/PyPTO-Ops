# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""QK L2-normalization + shared sigmoid gate — nq = f32(q) * rsqrt(sum(f32(q)^2) + eps) row-wise; bf16 storage, fp32 arithmetic (fp32 path kept)."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


RESIDENT_ENABLED = True
ROW_TILE = 8
RESIDENT_ELEMENTS = 8192
COL_TILE = 256
STREAMED_ELEMENTS = 8192
EPS = 1e-6


def resolve_geometry(shape):
    rows = shape["M"]
    cols = shape["N"]
    if rows <= 0 or rows % 64 or cols <= 0 or cols % 128:
        raise ValueError("Expected positive M%64==0 and N%128==0")
    if RESIDENT_ENABLED and ROW_TILE * cols <= RESIDENT_ELEMENTS:
        return "resident", ROW_TILE, cols
    col_tile = COL_TILE
    while col_tile > cols or cols % col_tile:
        col_tile //= 2
    return "streamed", 8, col_tile


def _streamed_program(rows, cols, row_tile, col_tile, storage_type):
    @pl.program
    class QKSharedGateStreamed:
        @pl.function
        def main(
            self,
            q: pl.Tensor[[rows, cols], storage_type],
            k: pl.Tensor[[rows, cols], storage_type],
            g: pl.Tensor[[rows, cols], storage_type],
            stat_q: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            stat_k: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            nq: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            nk: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            sg: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            yq: pl.Out[pl.Tensor[[rows, cols], storage_type]],
            yk: pl.Out[pl.Tensor[[rows, cols], storage_type]],
        ):
            for row_start in pl.parallel(0, rows, row_tile):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_gate_streamed"):
                    q_sum = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                    k_sum = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                    for col_start in pl.range(0, cols, col_tile):
                        q_tile = pl.cast(q[row_start:row_start + row_tile, col_start:col_start + col_tile], target_type=pl.FP32)
                        k_tile = pl.cast(k[row_start:row_start + row_tile, col_start:col_start + col_tile], target_type=pl.FP32)
                        q_partial = pl.reshape(pl.row_sum(pl.mul(q_tile, q_tile)), [1, row_tile])
                        k_partial = pl.reshape(pl.row_sum(pl.mul(k_tile, k_tile)), [1, row_tile])
                        q_sum = pl.add(q_sum, q_partial)
                        k_sum = pl.add(k_sum, k_partial)
                    stat_q[:, row_start:row_start + row_tile] = q_sum
                    stat_k[:, row_start:row_start + row_tile] = k_sum
                    q_inverse = pl.rsqrt(pl.add(q_sum, EPS), high_precision=True)
                    k_inverse = pl.rsqrt(pl.add(k_sum, EPS), high_precision=True)
                    q_inverse_col = pl.reshape(q_inverse, [row_tile, 1])
                    k_inverse_col = pl.reshape(k_inverse, [row_tile, 1])
                    for col_start in pl.range(0, cols, col_tile):
                        q_tile = pl.cast(q[row_start:row_start + row_tile, col_start:col_start + col_tile], target_type=pl.FP32)
                        k_tile = pl.cast(k[row_start:row_start + row_tile, col_start:col_start + col_tile], target_type=pl.FP32)
                        gate_tile = pl.cast(g[row_start:row_start + row_tile, col_start:col_start + col_tile], target_type=pl.FP32)
                        q_normalized = pl.row_expand_mul(q_tile, q_inverse_col)
                        k_normalized = pl.row_expand_mul(k_tile, k_inverse_col)
                        gate_denominator = pl.add(pl.exp(pl.neg(gate_tile)), 1.0)
                        gate = pl.div(pl.full([row_tile, col_tile], dtype=pl.FP32, value=1.0), gate_denominator)
                        nq[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.cast(q_normalized, target_type=storage_type, mode="rint")
                        nk[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.cast(k_normalized, target_type=storage_type, mode="rint")
                        sg[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.cast(gate, target_type=storage_type, mode="rint")
                        yq[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.cast(pl.mul(q_normalized, gate), target_type=storage_type, mode="rint")
                        yk[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.cast(pl.mul(k_normalized, gate), target_type=storage_type, mode="rint")
            return stat_q, stat_k, nq, nk, sg, yq, yk

    return QKSharedGateStreamed


def _streamed_fp32_program(rows, cols, row_tile, col_tile):
    @pl.program
    class QKSharedGateStreamedFP32:
        @pl.function
        def main(
            self,
            q: pl.Tensor[[rows, cols], pl.FP32],
            k: pl.Tensor[[rows, cols], pl.FP32],
            g: pl.Tensor[[rows, cols], pl.FP32],
            stat_q: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            stat_k: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            nq: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            nk: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            sg: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            yq: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            yk: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ):
            for row_start in pl.parallel(0, rows, row_tile):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_gate_streamed"):
                    q_sum = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                    k_sum = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                    for col_start in pl.range(0, cols, col_tile):
                        q_tile = q[row_start:row_start + row_tile, col_start:col_start + col_tile]
                        k_tile = k[row_start:row_start + row_tile, col_start:col_start + col_tile]
                        q_partial = pl.reshape(pl.row_sum(pl.mul(q_tile, q_tile)), [1, row_tile])
                        k_partial = pl.reshape(pl.row_sum(pl.mul(k_tile, k_tile)), [1, row_tile])
                        q_sum = pl.add(q_sum, q_partial)
                        k_sum = pl.add(k_sum, k_partial)
                    stat_q[:, row_start:row_start + row_tile] = q_sum
                    stat_k[:, row_start:row_start + row_tile] = k_sum
                    q_inverse = pl.rsqrt(pl.add(q_sum, EPS), high_precision=True)
                    k_inverse = pl.rsqrt(pl.add(k_sum, EPS), high_precision=True)
                    q_inverse_col = pl.reshape(q_inverse, [row_tile, 1])
                    k_inverse_col = pl.reshape(k_inverse, [row_tile, 1])
                    for col_start in pl.range(0, cols, col_tile):
                        q_tile = q[row_start:row_start + row_tile, col_start:col_start + col_tile]
                        k_tile = k[row_start:row_start + row_tile, col_start:col_start + col_tile]
                        gate_tile = g[row_start:row_start + row_tile, col_start:col_start + col_tile]
                        q_normalized = pl.row_expand_mul(q_tile, q_inverse_col)
                        k_normalized = pl.row_expand_mul(k_tile, k_inverse_col)
                        gate_denominator = pl.add(pl.exp(pl.neg(gate_tile)), 1.0)
                        gate = pl.div(pl.full([row_tile, col_tile], dtype=pl.FP32, value=1.0), gate_denominator)
                        nq[row_start:row_start + row_tile, col_start:col_start + col_tile] = q_normalized
                        nk[row_start:row_start + row_tile, col_start:col_start + col_tile] = k_normalized
                        sg[row_start:row_start + row_tile, col_start:col_start + col_tile] = gate
                        yq[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.mul(q_normalized, gate)
                        yk[row_start:row_start + row_tile, col_start:col_start + col_tile] = pl.mul(k_normalized, gate)
            return stat_q, stat_k, nq, nk, sg, yq, yk

    return QKSharedGateStreamedFP32


SPLIT_OWNER = 0
BOUNDED_EPILOGUE = 0
PIPELINE_STAGE = 0




def _spmd_bf16_program(rows, cols, row_tile):
    window = 256
    while window > cols or cols % window:
        window //= 2

    @pl.program
    class QKSingleDispatchBF16:
        @pl.function
        def main(
            self,
            q: pl.Tensor[[rows, cols], pl.BF16],
            k: pl.Tensor[[rows, cols], pl.BF16],
            g: pl.Tensor[[rows, cols], pl.BF16],
            stat_q: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            stat_k: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            nq: pl.Out[pl.Tensor[[rows, cols], pl.BF16]],
            nk: pl.Out[pl.Tensor[[rows, cols], pl.BF16]],
            sg: pl.Out[pl.Tensor[[rows, cols], pl.BF16]],
            yq: pl.Out[pl.Tensor[[rows, cols], pl.BF16]],
            yk: pl.Out[pl.Tensor[[rows, cols], pl.BF16]],
        ):
            for task in pl.spmd(1 * (rows // row_tile), name_hint="qk_single_dispatch"):
                row_start = (task // 1) * row_tile
                q_tile = pl.cast(q[row_start:row_start + row_tile, :], target_type=pl.FP32)
                q_sum = pl.row_sum(pl.mul(q_tile, q_tile))
                q_sum_row = pl.reshape(q_sum, [1, row_tile])
                stat_q[:, row_start:row_start + row_tile] = q_sum_row
                q_inverse = pl.rsqrt(pl.add(q_sum_row, EPS), high_precision=True)
                q_inverse_col = pl.reshape(q_inverse, [row_tile, 1])
                q_normalized = pl.row_expand_mul(q_tile, q_inverse_col)
                nq[row_start:row_start + row_tile, :] = pl.cast(q_normalized, target_type=pl.BF16, mode="rint")
                k_tile = pl.cast(k[row_start:row_start + row_tile, :], target_type=pl.FP32)
                k_sum = pl.row_sum(pl.mul(k_tile, k_tile))
                k_sum_row = pl.reshape(k_sum, [1, row_tile])
                stat_k[:, row_start:row_start + row_tile] = k_sum_row
                k_inverse = pl.rsqrt(pl.add(k_sum_row, EPS), high_precision=True)
                k_inverse_col = pl.reshape(k_inverse, [row_tile, 1])
                k_normalized = pl.row_expand_mul(k_tile, k_inverse_col)
                nk[row_start:row_start + row_tile, :] = pl.cast(k_normalized, target_type=pl.BF16, mode="rint")
                gate_tile = pl.cast(g[row_start:row_start + row_tile, :], target_type=pl.FP32)
                gate_denominator = pl.add(pl.exp(pl.neg(gate_tile)), 1.0)
                gate = pl.div(pl.full([row_tile, cols], dtype=pl.FP32, value=1.0), gate_denominator)
                sg[row_start:row_start + row_tile, :] = pl.cast(gate, target_type=pl.BF16, mode="rint")
                yq[row_start:row_start + row_tile, :] = pl.cast(pl.mul(q_normalized, gate), target_type=pl.BF16, mode="rint")
                yk[row_start:row_start + row_tile, :] = pl.cast(pl.mul(k_normalized, gate), target_type=pl.BF16, mode="rint")
            return stat_q, stat_k, nq, nk, sg, yq, yk

    return QKSingleDispatchBF16

def _spmd_fp32_program(rows, cols, row_tile):
    window = 256
    while window > cols or cols % window:
        window //= 2

    @pl.program
    class QKSingleDispatchFP32:
        @pl.function
        def main(
            self,
            q: pl.Tensor[[rows, cols], pl.FP32],
            k: pl.Tensor[[rows, cols], pl.FP32],
            g: pl.Tensor[[rows, cols], pl.FP32],
            stat_q: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            stat_k: pl.Out[pl.Tensor[[1, rows], pl.FP32]],
            nq: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            nk: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            sg: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            yq: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            yk: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ):
            for task in pl.spmd(1 * (rows // row_tile), name_hint="qk_single_dispatch"):
                row_start = (task // 1) * row_tile
                q_tile = q[row_start:row_start + row_tile, :]
                q_sum = pl.row_sum(pl.mul(q_tile, q_tile))
                q_sum_row = pl.reshape(q_sum, [1, row_tile])
                stat_q[:, row_start:row_start + row_tile] = q_sum_row
                q_inverse = pl.rsqrt(pl.add(q_sum_row, EPS), high_precision=True)
                q_inverse_col = pl.reshape(q_inverse, [row_tile, 1])
                q_normalized = pl.row_expand_mul(q_tile, q_inverse_col)
                nq[row_start:row_start + row_tile, :] = q_normalized
                k_tile = k[row_start:row_start + row_tile, :]
                k_sum = pl.row_sum(pl.mul(k_tile, k_tile))
                k_sum_row = pl.reshape(k_sum, [1, row_tile])
                stat_k[:, row_start:row_start + row_tile] = k_sum_row
                k_inverse = pl.rsqrt(pl.add(k_sum_row, EPS), high_precision=True)
                k_inverse_col = pl.reshape(k_inverse, [row_tile, 1])
                k_normalized = pl.row_expand_mul(k_tile, k_inverse_col)
                nk[row_start:row_start + row_tile, :] = k_normalized
                gate_tile = g[row_start:row_start + row_tile, :]
                gate_denominator = pl.add(pl.exp(pl.neg(gate_tile)), 1.0)
                gate = pl.div(pl.full([row_tile, cols], dtype=pl.FP32, value=1.0), gate_denominator)
                sg[row_start:row_start + row_tile, :] = gate
                yq[row_start:row_start + row_tile, :] = pl.mul(q_normalized, gate)
                yk[row_start:row_start + row_tile, :] = pl.mul(k_normalized, gate)
            return stat_q, stat_k, nq, nk, sg, yq, yk

    return QKSingleDispatchFP32

def build(shape):
    import torch
    from golden import TensorSpec

    rows = shape["M"]
    cols = shape["N"]
    dtype = shape.get("dtype", "fp32")
    if dtype not in ("fp32", "bf16"):
        raise ValueError("Unsupported dtype")
    storage_type = pl.BF16 if dtype == "bf16" else pl.FP32
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    implementation, row_tile, col_tile = resolve_geometry(shape)
    if dtype == "bf16":
        if implementation == "resident":
            program = _spmd_bf16_program(rows, cols, row_tile)
        else:
            program = _streamed_program(rows, cols, row_tile, col_tile, storage_type)
    elif implementation == "resident":
        program = _spmd_fp32_program(rows, cols, row_tile)
    else:
        program = _streamed_fp32_program(rows, cols, row_tile, col_tile)
    specs = [
        TensorSpec("q", [rows, cols], torch_dtype, init_value=torch.randn),
        TensorSpec("k", [rows, cols], torch_dtype, init_value=torch.randn),
        TensorSpec("g", [rows, cols], torch_dtype, init_value=torch.randn),
        TensorSpec("stat_q", [1, rows], torch.float32),
        TensorSpec("stat_k", [1, rows], torch.float32),
        TensorSpec("nq", [rows, cols], torch_dtype),
        TensorSpec("nk", [rows, cols], torch_dtype),
        TensorSpec("sg", [rows, cols], torch_dtype),
        TensorSpec("yq", [rows, cols], torch_dtype),
        TensorSpec("yk", [rows, cols], torch_dtype),
    ]

    def golden(tensors):
        q_float = tensors["q"].float()
        k_float = tensors["k"].float()
        g_float = tensors["g"].float()
        q_sum = q_float.square().sum(dim=-1, keepdim=True)
        k_sum = k_float.square().sum(dim=-1, keepdim=True)
        q_normalized = q_float * torch.rsqrt(q_sum + EPS)
        k_normalized = k_float * torch.rsqrt(k_sum + EPS)
        gate = torch.sigmoid(g_float)
        tensors["stat_q"][:] = q_sum.reshape(1, -1)
        tensors["stat_k"][:] = k_sum.reshape(1, -1)
        tensors["nq"][:] = q_normalized
        tensors["nk"][:] = k_normalized
        tensors["sg"][:] = gate
        tensors["yq"][:] = q_normalized * gate
        tensors["yk"][:] = k_normalized * gate

    tolerance = 2e-2 if dtype == "bf16" else 1e-4
    return program, specs, golden, tolerance, tolerance

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
