# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MoE permute/dispatch — exact, stable, dropless token routing (K=8, E=64): permuted x/w, counts, offsets and the inverse map."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


K = 8
E = 64
ROUTE_TILE = 128
OUTPUT_TILE = 128
COL_TILE = 2048
TOKEN_TILE = 4
INVERSE_PAD = 32
HIST_ROWS = 32
PAYLOAD_ROWS = 16
FP32_PAYLOAD_ROWS = 16
TOKENS = pl.dynamic("permute_tokens")
HIDDEN = pl.dynamic("permute_hidden")
ROUTES = pl.dynamic("permute_routes")


@pl.jit
def moe_permute_blocked_inverse(
    x: pl.Tensor[[TOKENS, HIDDEN], pl.FP32],
    route_indices: pl.Tensor[[TOKENS, K], pl.INT32],
    route_weights: pl.Tensor[[TOKENS, K], pl.FP32],
    counts: pl.InOut[pl.Tensor[[E], pl.INT32]],
    offsets: pl.InOut[pl.Tensor[[E + 1], pl.INT32]],
    dst_map: pl.InOut[pl.Tensor[[ROUTES], pl.INT32]],
    permuted_w: pl.Out[pl.Tensor[[ROUTES, 1], pl.FP32]],
    permuted_x: pl.Out[pl.Tensor[[ROUTES, HIDDEN], pl.FP32]],
):
    num_tokens = pl.tensor.dim(x, 0)
    num_columns = pl.tensor.dim(x, 1)
    num_routes = num_tokens * K
    num_blocks = (num_routes + ROUTE_TILE - 1) // ROUTE_TILE
    histograms = pl.create_tensor([num_blocks, E], dtype=pl.INT32, manual_dep=True)
    weight_packets = pl.create_tensor([num_routes, INVERSE_PAD], dtype=pl.INT32, manual_dep=True)
    flat_counts = pl.reshape(counts, [1, E])
    flat_offsets = pl.reshape(offsets, [1, E + 1])
    flat_map = pl.reshape(dst_map, [1, num_routes])
    flat_packets = pl.reshape(weight_packets, [1, num_routes * INVERSE_PAD])
    flat_weights = pl.reshape(permuted_w, [1, num_routes])
    with pl.spmd(num_blocks, name_hint="blocked_count", allow_early_resolve=True) as count_tid:
        block = pl.tile.get_block_idx()
        token_begin = block * (ROUTE_TILE // K)
        token_rows = pl.min(ROUTE_TILE // K, num_tokens - token_begin)
        index_tile = pl.tile.load(route_indices, [token_begin, 0], [ROUTE_TILE // K, K], valid_shape=[token_rows, K])
        histogram = pl.array.create(E, pl.INT32)
        for expert in pl.range(E):
            histogram[expert] = 0
        for token_lane in pl.range(token_rows):
            for route in pl.range(K):
                expert = pl.cast(pl.tile.read(index_tile, [token_lane, route]), pl.INDEX)
                histogram[expert] = histogram[expert] + 1
        histogram_row = pl.tile.full([1, E], dtype=pl.INT32, value=0)
        for expert in pl.range(E):
            pl.tile.write(histogram_row, [0, expert], histogram[expert])
        pl.tile.store(histogram_row, [block, 0], histograms)
    with pl.spmd(num_blocks, name_hint="histogram_consumer_map", deps=[count_tid], allow_early_resolve=True) as map_tid:
        block = pl.tile.get_block_idx()
        token_begin = block * (ROUTE_TILE // K)
        token_rows = pl.min(ROUTE_TILE // K, num_tokens - token_begin)
        index_tile = pl.tile.load(route_indices, [token_begin, 0], [ROUTE_TILE // K, K], valid_shape=[token_rows, K])
        totals = pl.tile.full([1, E], dtype=pl.INT32, value=0)
        preceding = pl.tile.full([1, E], dtype=pl.INT32, value=0)
        for histogram_begin in pl.range(0, num_blocks, HIST_ROWS):
            histogram_rows = pl.min(HIST_ROWS, num_blocks - histogram_begin)
            histogram_tile = pl.tile.load(histograms, [histogram_begin, 0], [HIST_ROWS, E], valid_shape=[histogram_rows, E])
            totals = pl.tile.add(totals, pl.tile.col_sum(histogram_tile))
            if block > histogram_begin:
                prior_rows = pl.min(histogram_rows, block - histogram_begin)
                prior_tile = pl.tile.slice(histogram_tile, [HIST_ROWS, E], [0, 0], valid_shape=[prior_rows, E])
                preceding = pl.tile.add(preceding, pl.tile.col_sum(prior_tile))
        cursor = pl.array.create(E, pl.INT32)
        offsets_tile = pl.tile.full([1, 72], dtype=pl.INT32, value=0)
        running = pl.cast(0, pl.INT32)
        for expert in pl.range(E):
            cursor[expert] = running + pl.tile.read(preceding, [0, expert])
            if block == 0:
                pl.tile.write(offsets_tile, [0, expert], running)
            running = running + pl.tile.read(totals, [0, expert])
        if block == 0:
            pl.tile.store(totals, [0, 0], flat_counts)
            pl.tile.write(offsets_tile, [0, E], running)
            valid_offsets = pl.set_validshape(offsets_tile, 1, E + 1)
            pl.tile.store(valid_offsets, [0, 0], flat_offsets)
        map_tile = pl.tile.full([1, ROUTE_TILE], dtype=pl.INT32, value=0)
        weight_packet = pl.tile.full([1, INVERSE_PAD], dtype=pl.INT32, value=0)
        source_weights = pl.tile.load(route_weights, [token_begin, 0], [ROUTE_TILE // K, K], valid_shape=[token_rows, K])
        weight_bits = pl.tile.reinterpret_view(source_weights, pl.INT32)
        for token_lane in pl.range(token_rows):
            for route in pl.range(K):
                expert = pl.cast(pl.tile.read(index_tile, [token_lane, route]), pl.INDEX)
                slot = cursor[expert]
                cursor[expert] = slot + 1
                pl.tile.write(map_tile, [0, token_lane * K + route], slot)
                pl.tile.write(weight_packet, [0, 0], pl.cast(token_begin + token_lane, pl.INT32))
                pl.tile.write(weight_packet, [0, 1], pl.tile.read(weight_bits, [token_lane, route]))
                pl.tile.store(weight_packet, [pl.cast(slot, pl.INDEX), 0], weight_packets)
        valid_map = pl.set_validshape(map_tile, 1, token_rows * K)
        pl.tile.store(valid_map, [0, token_begin * K], flat_map)
    with pl.spmd((num_routes + OUTPUT_TILE - 1) // OUTPUT_TILE, name_hint="output_owned_materialize", deps=[map_tid]) as payload_tid:
        output_begin = pl.tile.get_block_idx() * OUTPUT_TILE
        output_rows = pl.min(OUTPUT_TILE, num_routes - output_begin)
        packet_row = pl.tile.load(flat_packets, [0, output_begin * INVERSE_PAD], [1, OUTPUT_TILE * INVERSE_PAD], valid_shape=[1, output_rows * INVERSE_PAD])
        packet_offsets = pl.tile.muls(pl.tile.arange(0, [1, OUTPUT_TILE], dtype=pl.INT32), INVERSE_PAD)
        gather_workspace = pl.tile.full([1, OUTPUT_TILE], dtype=pl.INT32, value=0)
        token_ids = pl.tile.gather(packet_row, packet_offsets, gather_workspace)
        weight_offsets = pl.tile.adds(packet_offsets, 1)
        weight_workspace = pl.tile.full([1, OUTPUT_TILE], dtype=pl.INT32, value=0)
        weight_bits_tile = pl.tile.gather(packet_row, weight_offsets, weight_workspace)
        weight_tile = pl.tile.reinterpret_view(weight_bits_tile, pl.FP32)
        valid_weights = pl.set_validshape(weight_tile, 1, output_rows)
        pl.tile.store(valid_weights, [0, output_begin], flat_weights)
        for row_begin in pl.range(0, output_rows, FP32_PAYLOAD_ROWS):
            valid_rows = pl.min(FP32_PAYLOAD_ROWS, output_rows - row_begin)
            for column in pl.range(0, num_columns, COL_TILE):
                valid_columns = pl.min(COL_TILE, num_columns - column)
                payload_tile = pl.tile.create([FP32_PAYLOAD_ROWS, COL_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                for row_lane in pl.range(valid_rows):
                    source_token = pl.cast(pl.tile.read(token_ids, [0, row_begin + row_lane]), pl.INDEX)
                    payload_tile = pl.tile.gather_row(payload_tile, x, [row_lane, 0], [source_token, column], [1, COL_TILE], valid_shape=[1, valid_columns])
                valid_payload = pl.set_validshape(payload_tile, valid_rows, valid_columns)
                pl.tile.store(valid_payload, [output_begin + row_begin, column], permuted_x)
    return permuted_x


@pl.jit
def moe_permute_blocked_inverse_bf16(
    x: pl.Tensor[[TOKENS, HIDDEN], pl.BF16],
    route_indices: pl.Tensor[[TOKENS, K], pl.INT32],
    route_weights: pl.Tensor[[TOKENS, K], pl.FP32],
    counts: pl.InOut[pl.Tensor[[E], pl.INT32]],
    offsets: pl.InOut[pl.Tensor[[E + 1], pl.INT32]],
    dst_map: pl.InOut[pl.Tensor[[ROUTES], pl.INT32]],
    permuted_w: pl.Out[pl.Tensor[[ROUTES, 1], pl.FP32]],
    permuted_x: pl.Out[pl.Tensor[[ROUTES, HIDDEN], pl.BF16]],
):
    num_tokens = pl.tensor.dim(x, 0)
    num_columns = pl.tensor.dim(x, 1)
    num_routes = num_tokens * K
    num_blocks = (num_routes + ROUTE_TILE - 1) // ROUTE_TILE
    histograms = pl.create_tensor([num_blocks, E], dtype=pl.INT32, manual_dep=True)
    weight_packets = pl.create_tensor([num_routes, INVERSE_PAD], dtype=pl.INT32, manual_dep=True)
    flat_counts = pl.reshape(counts, [1, E])
    flat_offsets = pl.reshape(offsets, [1, E + 1])
    flat_map = pl.reshape(dst_map, [1, num_routes])
    flat_packets = pl.reshape(weight_packets, [1, num_routes * INVERSE_PAD])
    flat_weights = pl.reshape(permuted_w, [1, num_routes])
    with pl.spmd(num_blocks, name_hint="blocked_count", allow_early_resolve=True) as count_tid:
        block = pl.tile.get_block_idx()
        token_begin = block * (ROUTE_TILE // K)
        token_rows = pl.min(ROUTE_TILE // K, num_tokens - token_begin)
        index_tile = pl.tile.load(route_indices, [token_begin, 0], [ROUTE_TILE // K, K], valid_shape=[token_rows, K])
        histogram = pl.array.create(E, pl.INT32)
        for expert in pl.range(E):
            histogram[expert] = 0
        for token_lane in pl.range(token_rows):
            for route in pl.range(K):
                expert = pl.cast(pl.tile.read(index_tile, [token_lane, route]), pl.INDEX)
                histogram[expert] = histogram[expert] + 1
        histogram_row = pl.tile.full([1, E], dtype=pl.INT32, value=0)
        for expert in pl.range(E):
            pl.tile.write(histogram_row, [0, expert], histogram[expert])
        pl.tile.store(histogram_row, [block, 0], histograms)
    with pl.spmd(num_blocks, name_hint="histogram_consumer_map", deps=[count_tid], allow_early_resolve=True) as map_tid:
        block = pl.tile.get_block_idx()
        token_begin = block * (ROUTE_TILE // K)
        token_rows = pl.min(ROUTE_TILE // K, num_tokens - token_begin)
        index_tile = pl.tile.load(route_indices, [token_begin, 0], [ROUTE_TILE // K, K], valid_shape=[token_rows, K])
        totals = pl.tile.full([1, E], dtype=pl.INT32, value=0)
        preceding = pl.tile.full([1, E], dtype=pl.INT32, value=0)
        for histogram_begin in pl.range(0, num_blocks, HIST_ROWS):
            histogram_rows = pl.min(HIST_ROWS, num_blocks - histogram_begin)
            histogram_tile = pl.tile.load(histograms, [histogram_begin, 0], [HIST_ROWS, E], valid_shape=[histogram_rows, E])
            totals = pl.tile.add(totals, pl.tile.col_sum(histogram_tile))
            if block > histogram_begin:
                prior_rows = pl.min(histogram_rows, block - histogram_begin)
                prior_tile = pl.tile.slice(histogram_tile, [HIST_ROWS, E], [0, 0], valid_shape=[prior_rows, E])
                preceding = pl.tile.add(preceding, pl.tile.col_sum(prior_tile))
        cursor = pl.array.create(E, pl.INT32)
        offsets_tile = pl.tile.full([1, 72], dtype=pl.INT32, value=0)
        running = pl.cast(0, pl.INT32)
        for expert in pl.range(E):
            cursor[expert] = running + pl.tile.read(preceding, [0, expert])
            if block == 0:
                pl.tile.write(offsets_tile, [0, expert], running)
            running = running + pl.tile.read(totals, [0, expert])
        if block == 0:
            pl.tile.store(totals, [0, 0], flat_counts)
            pl.tile.write(offsets_tile, [0, E], running)
            valid_offsets = pl.set_validshape(offsets_tile, 1, E + 1)
            pl.tile.store(valid_offsets, [0, 0], flat_offsets)
        map_tile = pl.tile.full([1, ROUTE_TILE], dtype=pl.INT32, value=0)
        weight_packet = pl.tile.full([1, INVERSE_PAD], dtype=pl.INT32, value=0)
        source_weights = pl.tile.load(route_weights, [token_begin, 0], [ROUTE_TILE // K, K], valid_shape=[token_rows, K])
        weight_bits = pl.tile.reinterpret_view(source_weights, pl.INT32)
        for token_lane in pl.range(token_rows):
            for route in pl.range(K):
                expert = pl.cast(pl.tile.read(index_tile, [token_lane, route]), pl.INDEX)
                slot = cursor[expert]
                cursor[expert] = slot + 1
                pl.tile.write(map_tile, [0, token_lane * K + route], slot)
                pl.tile.write(weight_packet, [0, 0], pl.cast(token_begin + token_lane, pl.INT32))
                pl.tile.write(weight_packet, [0, 1], pl.tile.read(weight_bits, [token_lane, route]))
                pl.tile.store(weight_packet, [pl.cast(slot, pl.INDEX), 0], weight_packets)
        valid_map = pl.set_validshape(map_tile, 1, token_rows * K)
        pl.tile.store(valid_map, [0, token_begin * K], flat_map)
    with pl.spmd((num_routes + OUTPUT_TILE - 1) // OUTPUT_TILE, name_hint="output_owned_materialize", deps=[map_tid]) as payload_tid:
        output_begin = pl.tile.get_block_idx() * OUTPUT_TILE
        output_rows = pl.min(OUTPUT_TILE, num_routes - output_begin)
        packet_row = pl.tile.load(flat_packets, [0, output_begin * INVERSE_PAD], [1, OUTPUT_TILE * INVERSE_PAD], valid_shape=[1, output_rows * INVERSE_PAD])
        packet_offsets = pl.tile.muls(pl.tile.arange(0, [1, OUTPUT_TILE], dtype=pl.INT32), INVERSE_PAD)
        gather_workspace = pl.tile.full([1, OUTPUT_TILE], dtype=pl.INT32, value=0)
        token_ids = pl.tile.gather(packet_row, packet_offsets, gather_workspace)
        weight_offsets = pl.tile.adds(packet_offsets, 1)
        weight_workspace = pl.tile.full([1, OUTPUT_TILE], dtype=pl.INT32, value=0)
        weight_bits_tile = pl.tile.gather(packet_row, weight_offsets, weight_workspace)
        weight_tile = pl.tile.reinterpret_view(weight_bits_tile, pl.FP32)
        valid_weights = pl.set_validshape(weight_tile, 1, output_rows)
        pl.tile.store(valid_weights, [0, output_begin], flat_weights)
        for row_begin in pl.range(0, output_rows, PAYLOAD_ROWS):
            valid_rows = pl.min(PAYLOAD_ROWS, output_rows - row_begin)
            for column in pl.range(0, num_columns, COL_TILE):
                valid_columns = pl.min(COL_TILE, num_columns - column)
                payload_tile = pl.tile.create([PAYLOAD_ROWS, COL_TILE], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                for row_lane in pl.range(valid_rows):
                    source_token = pl.cast(pl.tile.read(token_ids, [0, row_begin + row_lane]), pl.INDEX)
                    payload_tile = pl.tile.gather_row(payload_tile, x, [row_lane, 0], [source_token, column], [1, COL_TILE], valid_shape=[1, valid_columns])
                valid_payload = pl.set_validshape(payload_tile, valid_rows, valid_columns)
                pl.tile.store(valid_payload, [output_begin + row_begin, column], permuted_x)
    return permuted_x


def assert_dtype_contract(shape, specs, rtol, atol):
    import torch

    activation_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[shape["dtype"]]
    by_name = {spec.name: spec for spec in specs}
    for name in ("x", "permuted_x"):
        assert by_name[name].dtype == activation_dtype, (name, "activation dtype")
    for name in ("route_weights", "permuted_w"):
        assert by_name[name].dtype == torch.float32, (name, "weight dtype")
    for name in ("route_indices", "counts", "offsets", "dst_map"):
        assert by_name[name].dtype == torch.int32, (name, "metadata dtype")
    assert rtol == 0.0 and atol == 0.0

def build(shape: dict):
    import torch
    from golden import TensorSpec

    activation_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[shape["dtype"]]
    num_tokens = shape["M"]
    num_columns = shape["N"]
    num_routes = num_tokens * K
    specs = [
        TensorSpec("x", [num_tokens, num_columns], activation_dtype, init_value=torch.randn),
        TensorSpec("route_indices", [num_tokens, K], torch.int32,
                   init_value=lambda: torch.randint(0, E, (num_tokens, K), dtype=torch.int32)),
        TensorSpec("route_weights", [num_tokens, K], torch.float32, init_value=torch.randn),
        TensorSpec("counts", [E], torch.int32),
        TensorSpec("offsets", [E + 1], torch.int32),
        TensorSpec("dst_map", [num_routes], torch.int32),
        TensorSpec("permuted_w", [num_routes, 1], torch.float32),
        TensorSpec("permuted_x", [num_routes, num_columns], activation_dtype),
    ]

    def golden(tensors):
        flat = tensors["route_indices"].long().reshape(-1)
        histogram = torch.bincount(flat, minlength=E).to(torch.int32)
        expert_offsets = torch.zeros(E + 1, dtype=torch.int64)
        expert_offsets[1:] = torch.cumsum(histogram.long(), 0)
        order = torch.argsort(flat, stable=True)
        destination = torch.empty(num_routes, dtype=torch.int64)
        destination[order] = torch.arange(num_routes)
        tensors["counts"][:] = histogram
        tensors["offsets"][:] = expert_offsets.to(torch.int32)
        tensors["dst_map"][:] = destination.to(torch.int32)
        tensors["permuted_w"][:] = tensors["route_weights"].reshape(-1, 1)[order]
        tensors["permuted_x"][:] = tensors["x"][order // K]

    fn = moe_permute_blocked_inverse_bf16 if shape["dtype"] == "bf16" else moe_permute_blocked_inverse
    rtol = atol = 0.0
    assert_dtype_contract(shape, specs, rtol, atol)
    return fn, specs, golden, rtol, atol

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
