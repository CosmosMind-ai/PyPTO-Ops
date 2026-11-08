# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-owned weighted routes with independently packed workspace publication."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024}


MAP_TILE = 64
TOKEN_TILE = 16
MAP_LINE_WORDS = 16
K = 8


@pl.jit
def inverse_weighted_combine_1024(
    expert_x: pl.Tensor,
    row_w: pl.Tensor,
    src_idx: pl.Tensor,
    slot_x: pl.InOut[pl.Tensor],
    slot_w: pl.InOut[pl.Tensor],
    y: pl.Out[pl.Tensor],
):
    route_rows, hidden = expert_x.shape
    tokens = y.shape[0]
    inverse = pl.create_tensor([route_rows, MAP_LINE_WORDS], dtype=pl.INT32)
    source_slots = pl.reshape(src_idx, [1, route_rows])
    for row_base in pl.parallel(0, route_rows, MAP_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="inverse_map"):
            valid_rows = pl.min(MAP_TILE, route_rows - row_base)
            map_slots = pl.slice(
                source_slots, [1, MAP_TILE], [0, row_base],
                valid_shape=[1, valid_rows], clamp=True,
            )
            for row_offset in pl.range(MAP_TILE):
                row = row_base + row_offset
                if row < route_rows:
                    slot = pl.cast(pl.read(map_slots, [0, row_offset]), pl.INDEX)
                    pl.write(inverse, [slot, 0], pl.cast(row, pl.INT32))
    for token_base in pl.parallel(0, tokens, TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="weighted_combine"):
            output_weights = pl.full([1, TOKEN_TILE * K], dtype=pl.FP32, value=0.0)
            for token_offset in pl.range(TOKEN_TILE):
                token = token_base + token_offset
                if token < tokens:
                    for column in pl.range(0, hidden, 1024):
                        valid_columns = pl.min(1024, hidden - column)
                        accumulator = pl.full([1, 1024], dtype=pl.FP32, value=0.0)
                        packed_routes = pl.full([K, 1024], dtype=pl.FP32, value=0.0)
                        for route in pl.range(K):
                            slot = token * K + route
                            row = pl.cast(pl.read(inverse, [slot, 0]), pl.INDEX)
                            weight = pl.read(row_w, [row, 0])
                            values = pl.slice(
                                expert_x, [1, 1024], [row, column],
                                valid_shape=[1, valid_columns], clamp=True,
                            )
                            weighted = pl.mul(values, weight)
                            packed_routes = pl.assemble(packed_routes, weighted, [route, 0])
                            pl.write(output_weights, [0, token_offset * K + route], weight)
                            accumulator = pl.add(accumulator, weighted)
                        packed_routes = pl.set_validshape(packed_routes, K, valid_columns)
                        slot_x = pl.assemble(slot_x, packed_routes, [token * K, column])
                        accumulator = pl.set_validshape(accumulator, 1, valid_columns)
                        y = pl.assemble(y, accumulator, [token, column])
            valid_weights = pl.min(TOKEN_TILE, tokens - token_base) * K
            output_weights = pl.set_validshape(output_weights, 1, valid_weights)
            slot_w = pl.assemble(slot_w, output_weights, [0, token_base * K])
    return y


def build(shape):
    import torch
    from golden import TensorSpec

    tokens, hidden = shape["M"], shape["N"]
    route_rows = tokens * K
    specs = [
        TensorSpec("expert_x", [route_rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("row_w", [route_rows, 1], torch.float32, init_value=torch.randn),
        TensorSpec(
            "src_idx", [route_rows, 1], torch.int32,
            init_value=lambda: torch.randperm(route_rows, dtype=torch.int32).reshape(route_rows, 1),
        ),
        TensorSpec("slot_x", [route_rows, hidden], torch.float32),
        TensorSpec("slot_w", [1, route_rows], torch.float32),
        TensorSpec("y", [tokens, hidden], torch.float32),
    ]

    def golden(tensors):
        slots = tensors["src_idx"].long().reshape(-1)
        weighted = tensors["expert_x"] * tensors["row_w"]
        tensors["slot_x"][slots] = weighted
        tensors["slot_w"][0, slots] = tensors["row_w"][:, 0]
        tensors["y"][:] = tensors["slot_x"].reshape(tokens, K, hidden).sum(dim=1)

    return inverse_weighted_combine_1024, specs, golden, 1e-4, 1e-4

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
