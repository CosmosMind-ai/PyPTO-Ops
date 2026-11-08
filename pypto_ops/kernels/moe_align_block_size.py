# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stable expert-blocked routing through private counting-sort runs."""

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024}


PARTITIONS = 4
ROUTE_TILE = 128
OWNER_EXPERTS = 64
MERGE_MODE = 0
FILL_TILE = 512


def build(shape):
    import torch
    from golden import TensorSpec

    tokens = int(shape["M"])
    experts = int(shape["N"])
    if tokens < 1 or experts < 1:
        raise ValueError("M and N must be positive")
    routes = tokens * 8
    capacity = routes + experts * 63
    blocks = capacity // 64
    partitions = min(tokens, max(PARTITIONS, (tokens + 2047) // 2048))
    shard_rows = (tokens + partitions - 1) // partitions
    shard_stride = ((shard_rows * 8 + 15) // 16) * 16
    expert_stride = ((experts + 15) // 16) * 16
    label_stride = ((blocks + 15) // 16) * 16

    @pl.program
    class StableRunsExpertOwnedMerge:
        @pl.function
        def main(
            self,
            topk_ids: pl.Tensor[[tokens, 8], pl.INT32],
            counts: pl.InOut[pl.Tensor[[experts], pl.INT32]],
            expert_starts: pl.InOut[pl.Tensor[[experts], pl.INT32]],
            sorted_token_ids: pl.Out[pl.Tensor[[1, capacity], pl.INT32]],
            expert_ids: pl.Out[pl.Tensor[[1, blocks], pl.INT32]],
            num_tokens_post_pad: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fill_sorted") as filled:
                for fill_base in pl.range(0, capacity, FILL_TILE):
                    sentinel = pl.tile.full([1, FILL_TILE], dtype=pl.INT32, value=routes)
                    sentinel_valid = pl.set_validshape(sentinel, 1, pl.min(FILL_TILE, capacity - fill_base))
                    pl.store(sentinel_valid, [0, fill_base], sorted_token_ids)

            histograms = pl.create_tensor([partitions, expert_stride], dtype=pl.INT32)
            local_starts = pl.create_tensor([partitions, expert_stride], dtype=pl.INT32)
            compact_runs = pl.create_tensor([partitions, shard_stride], dtype=pl.INT32)
            for partition in pl.parallel(partitions):
                histogram_row = pl.slice(histograms, [1, expert_stride], [partition, 0])
                local_start_row = pl.slice(local_starts, [1, expert_stride], [partition, 0])
                compact_row = pl.slice(compact_runs, [1, shard_stride], [partition, 0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="local_stable_runs"):
                    cursor = pl.tile.full([1, expert_stride], dtype=pl.INT32, value=0)
                    for row_base in pl.range(0, shard_rows, ROUTE_TILE // 8):
                        input_row = partition * shard_rows + row_base
                        valid_rows = pl.max(0, pl.min(ROUTE_TILE // 8, pl.min(shard_rows - row_base, tokens - input_row)))
                        if valid_rows > 0:
                            input_tile = pl.load(topk_ids, [input_row, 0], [ROUTE_TILE // 8, 8], valid_shape=[valid_rows, 8])
                            for token_offset in pl.range(valid_rows):
                                for choice in pl.range(8):
                                    counted_expert = pl.cast(pl.read(input_tile, [token_offset, choice]), pl.INDEX)
                                    local_count = pl.read(cursor, [0, counted_expert])
                                    pl.write(cursor, [0, counted_expert], pl.cast(local_count + 1, pl.INT32))
                    pl.store(cursor, [0, 0], histogram_row)
                    local_running = pl.cast(0, pl.INT32)
                    for prefix_expert in pl.range(experts):
                        prefix_count = pl.read(cursor, [0, prefix_expert])
                        pl.write(cursor, [0, prefix_expert], local_running)
                        local_running = local_running + prefix_count
                    pl.store(cursor, [0, 0], local_start_row)
                    compact = pl.tile.full([1, shard_stride], dtype=pl.INT32, value=routes)
                    for place_row_base in pl.range(0, shard_rows, ROUTE_TILE // 8):
                        place_input_row = partition * shard_rows + place_row_base
                        place_valid_rows = pl.max(0, pl.min(ROUTE_TILE // 8, pl.min(shard_rows - place_row_base, tokens - place_input_row)))
                        if place_valid_rows > 0:
                            place_input_tile = pl.load(topk_ids, [place_input_row, 0], [ROUTE_TILE // 8, 8], valid_shape=[place_valid_rows, 8])
                            for place_token_offset in pl.range(place_valid_rows):
                                for place_choice in pl.range(8):
                                    placed_expert = pl.cast(pl.read(place_input_tile, [place_token_offset, place_choice]), pl.INDEX)
                                    local_slot = pl.cast(pl.read(cursor, [0, placed_expert]), pl.INDEX)
                                    local_route = (place_input_row + place_token_offset) * 8 + place_choice
                                    pl.write(compact, [0, local_slot], pl.cast(local_route, pl.INT32))
                                    pl.write(cursor, [0, placed_expert], pl.cast(local_slot + 1, pl.INT32))
                    pl.store(compact, [0, 0], compact_row)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="expert_metadata"):
                totals = pl.tile.full([1, expert_stride], dtype=pl.INT32, value=0)
                for sum_partition in pl.range(partitions):
                    shard_hist = pl.load(histograms, [sum_partition, 0], [1, expert_stride])
                    totals = pl.add(totals, shard_hist)
                labels = pl.tile.full([1, label_stride], dtype=pl.INT32, value=-1)
                global_running = pl.cast(0, pl.INT32)
                for metadata_expert in pl.range(experts):
                    expert_count = pl.read(totals, [0, metadata_expert])
                    pl.write(counts, [metadata_expert], expert_count)
                    pl.write(expert_starts, [metadata_expert], global_running)
                    aligned_blocks = pl.cast((expert_count + 63) // 64, pl.INDEX)
                    first_block = pl.cast(global_running // 64, pl.INDEX)
                    for label_block in pl.range(aligned_blocks):
                        pl.write(labels, [0, first_block + label_block], pl.cast(metadata_expert, pl.INT32))
                    global_running = global_running + pl.cast(aligned_blocks * 64, pl.INT32)
                pl.store(pl.set_validshape(labels, 1, blocks), [0, 0], expert_ids)
                pl.write(num_tokens_post_pad, [0, 0], global_running)

            for expert_base in pl.parallel(0, experts, OWNER_EXPERTS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="expert_owned_merge", deps=[filled], no_dep_args=[sorted_token_ids]):
                    valid_experts = pl.min(OWNER_EXPERTS, experts - expert_base)
                    hist = pl.load(histograms, [0, expert_base], [partitions, OWNER_EXPERTS], valid_shape=[partitions, valid_experts])
                    starts = pl.load(local_starts, [0, expert_base], [partitions, OWNER_EXPERTS], valid_shape=[partitions, valid_experts])
                    if MERGE_MODE == 0:
                        for local_expert in pl.range(valid_experts):
                            destination = pl.cast(pl.read(expert_starts, [expert_base + local_expert]), pl.INDEX)
                            for merge_partition in pl.range(partitions):
                                span_count = pl.cast(pl.read(hist, [merge_partition, local_expert]), pl.INDEX)
                                source_start = pl.cast(pl.read(starts, [merge_partition, local_expert]), pl.INDEX)
                                for route_offset in pl.range(span_count):
                                    gathered_route = pl.read(compact_runs, [merge_partition, source_start + route_offset])
                                    pl.write(sorted_token_ids, [0, destination + route_offset], gathered_route)
                                destination = destination + span_count
                    else:
                        destinations = pl.tile.full([1, OWNER_EXPERTS], dtype=pl.INT32, value=0)
                        for cursor_expert in pl.range(valid_experts):
                            global_start = pl.read(expert_starts, [expert_base + cursor_expert])
                            pl.write(destinations, [0, cursor_expert], global_start)
                        for resident_partition in pl.range(partitions):
                            resident_run = pl.load(compact_runs, [resident_partition, 0], [1, shard_stride])
                            for resident_expert in pl.range(valid_experts):
                                resident_destination = pl.cast(pl.read(destinations, [0, resident_expert]), pl.INDEX)
                                resident_count = pl.cast(pl.read(hist, [resident_partition, resident_expert]), pl.INDEX)
                                resident_source = pl.cast(pl.read(starts, [resident_partition, resident_expert]), pl.INDEX)
                                for resident_offset in pl.range(resident_count):
                                    resident_route = pl.read(resident_run, [0, resident_source + resident_offset])
                                    pl.write(sorted_token_ids, [0, resident_destination + resident_offset], resident_route)
                                pl.write(destinations, [0, resident_expert], pl.cast(resident_destination + resident_count, pl.INT32))
            return sorted_token_ids

    def golden(tensors):
        flat_ids = tensors["topk_ids"].to(torch.int64).reshape(-1)
        expert_counts = torch.bincount(flat_ids, minlength=experts).to(torch.int32)
        aligned_counts = ((expert_counts + 63) // 64) * 64
        expert_offsets = torch.cumsum(aligned_counts, 0) - aligned_counts
        tensors["counts"].copy_(expert_counts)
        tensors["expert_starts"].copy_(expert_offsets.to(torch.int32))
        tensors["num_tokens_post_pad"].fill_(int(aligned_counts.sum()))
        sorted_routes = torch.full((capacity,), routes, dtype=torch.int32)
        labels = torch.full((blocks,), -1, dtype=torch.int32)
        for expert in range(experts):
            matching_routes = torch.nonzero(flat_ids == expert, as_tuple=True)[0].to(torch.int32)
            start = int(expert_offsets[expert])
            sorted_routes[start:start + matching_routes.numel()] = matching_routes
            labels[start // 64:(start + int(aligned_counts[expert])) // 64] = expert
        tensors["sorted_token_ids"].copy_(sorted_routes.reshape(1, capacity))
        tensors["expert_ids"].copy_(labels.reshape(1, blocks))

    specs = [
        TensorSpec("topk_ids", [tokens, 8], torch.int32, init_value=lambda: torch.randint(0, experts, (tokens, 8), dtype=torch.int32)),
        TensorSpec("counts", [experts], torch.int32),
        TensorSpec("expert_starts", [experts], torch.int32),
        TensorSpec("sorted_token_ids", [1, capacity], torch.int32),
        TensorSpec("expert_ids", [1, blocks], torch.int32),
        TensorSpec("num_tokens_post_pad", [1, 1], torch.int32),
    ]
    return StableRunsExpertOwnedMerge, specs, golden, 0.0, 0.0

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
