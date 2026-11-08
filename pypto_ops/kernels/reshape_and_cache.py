# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact KV scatter with independent complete-row selection ownership."""

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


HEAD_DIM = 128
STRIPE_WIDTH = 128
ROW_GROUP = 2


def make_program(token_rows, cache_rows, storage):
    sort_width = 32
    merge_stages = 0
    while sort_width < token_rows:
        sort_width *= 4
        merge_stages += 1
    owner_blocks = (cache_rows + STRIPE_WIDTH - 1) // STRIPE_WIDTH

    @pl.program
    class ReshapeAndCache:
        @pl.function(type=pl.FunctionType.InCore)
        def own_destinations(
            self,
            k: pl.Tensor[[token_rows, HEAD_DIM], storage],
            v: pl.Tensor[[token_rows, HEAD_DIM], storage],
            slot_mapping: pl.Tensor[[1, token_rows], pl.INT32],
            slotidx: pl.Tensor[[1, cache_rows], pl.FP32],
            sel: pl.InOut[pl.Tensor[[token_rows, cache_rows], pl.FP32]],
            hit: pl.InOut[pl.Tensor[[1, cache_rows], pl.FP32]],
            gathered_k: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
            gathered_v: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
            k_cache: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
            v_cache: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
        ) -> tuple[
            pl.Tensor[[token_rows, cache_rows], pl.FP32],
            pl.Tensor[[1, cache_rows], pl.FP32],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
        ]:
            stripe_start = pl.tile.get_block_idx() * STRIPE_WIDTH
            stripe_valid = pl.min(STRIPE_WIDTH, cache_rows - stripe_start)
            stripe_end = stripe_start + stripe_valid
            routing = pl.load(slot_mapping, [0, 0], [1, token_rows])
            for selection_start in pl.range(
                pl.tile.get_block_idx() * ROW_GROUP, token_rows, owner_blocks * ROW_GROUP
            ):
                selection_rows = pl.min(ROW_GROUP, token_rows - selection_start)
                selected_rows: pl.Tile[[ROW_GROUP, cache_rows], pl.FP32] = pl.tile.full(
                    [ROW_GROUP, cache_rows], dtype=pl.FP32, value=0.0
                )
                for local_row in pl.range(selection_rows):
                    selected_slot = pl.cast(pl.read(routing, [0, selection_start + local_row]), pl.INDEX)
                    pl.write(selected_rows, [local_row, selected_slot], 1.0)
                selected_tail = pl.tile.set_validshape(selected_rows, selection_rows, cache_rows)
                sel = pl.store(selected_tail, [selection_start, 0], sel)
            routing_fp32 = pl.cast(routing, pl.FP32)
            padded: pl.Tile[[1, sort_width], pl.FP32] = pl.tile.full(
                [1, sort_width], dtype=pl.FP32, value=-1.0
            )
            padded = pl.tile.assemble(padded, routing_fp32, [0, 0])
            token_ramp = pl.tile.arange(0, [1, sort_width], dtype=pl.INT32)
            token_ids = pl.tile.reinterpret_view(token_ramp, pl.UINT32)
            sorted_pairs = pl.tile.sort32(padded, token_ids)
            for merge_stage, (pairs,) in pl.range(merge_stages, init_values=(sorted_pairs,)):
                merged = pl.tile.mrgsort(pairs, block_len=1 << (6 + 2 * merge_stage))
                sorted_result = pl.yield_(merged)
            sorted_slots = pl.tile.gather_mask(sorted_result, mask_pattern=pl.tile.MaskPattern.P0101)
            sorted_tokens = pl.tile.gather_mask(
                sorted_result, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            sorted_slots_i32 = pl.cast(sorted_slots, pl.INT32)
            lower_threshold = pl.cast(pl.cast(stripe_start - 1, pl.INT32), pl.FP32)
            upper_threshold = pl.cast(pl.cast(stripe_end - 1, pl.INT32), pl.FP32)
            lower_flags = pl.tile.minimums(pl.tile.maximums(pl.sub(sorted_slots, lower_threshold), 0.0), 1.0)
            upper_flags = pl.tile.minimums(pl.tile.maximums(pl.sub(sorted_slots, upper_threshold), 0.0), 1.0)
            rank_flags: pl.Tile[[8, sort_width], pl.FP32] = pl.tile.full(
                [8, sort_width], dtype=pl.FP32, value=0.0
            )
            rank_flags = pl.tile.assemble(rank_flags, upper_flags, [0, 0])
            rank_flags = pl.tile.assemble(rank_flags, lower_flags, [1, 0])
            rank_flags_tail = pl.tile.set_validshape(rank_flags, 2, sort_width)
            rank_scratch: pl.Tile[[8, sort_width], pl.FP32] = pl.tile.create(
                [8, sort_width], dtype=pl.FP32
            )
            rank_counts = pl.tile.row_sum(rank_flags_tail, rank_scratch)
            rank_counts_i32 = pl.cast(rank_counts, pl.INT32)
            upper_total = pl.read(rank_counts_i32, [0, 0])
            lower_total = pl.read(rank_counts_i32, [1, 0])
            interval_start = pl.cast(upper_total, pl.INDEX)
            interval_end = pl.cast(lower_total, pl.INDEX)
            local_hit: pl.Tile[[1, STRIPE_WIDTH], pl.FP32] = pl.tile.full(
                [1, STRIPE_WIDTH], dtype=pl.FP32, value=0.0
            )
            gathered_zero: pl.Tile[[STRIPE_WIDTH, HEAD_DIM], storage] = pl.tile.full(
                [STRIPE_WIDTH, HEAD_DIM], dtype=storage, value=0.0
            )
            gathered_tail = pl.tile.set_validshape(gathered_zero, stripe_valid, HEAD_DIM)
            gathered_k = pl.store(gathered_tail, [stripe_start, 0], gathered_k)
            gathered_v = pl.store(gathered_tail, [stripe_start, 0], gathered_v)
            for position in pl.range(interval_start, interval_end):
                slot = pl.cast(pl.read(sorted_slots_i32, [0, position]), pl.INDEX)
                token = pl.cast(pl.read(sorted_tokens, [0, position]), pl.INDEX)
                local_slot = slot - stripe_start
                pl.write(local_hit, [0, local_slot], 1.0)
                key_row = pl.load(k, [token, 0], [1, HEAD_DIM])
                value_row = pl.load(v, [token, 0], [1, HEAD_DIM])
                k_cache = pl.store(key_row, [slot, 0], k_cache)
                v_cache = pl.store(value_row, [slot, 0], v_cache)
                gathered_k = pl.store(key_row, [slot, 0], gathered_k)
                gathered_v = pl.store(value_row, [slot, 0], gathered_v)
            hit_tail = pl.tile.set_validshape(local_hit, 1, stripe_valid)
            hit = pl.store(hit_tail, [0, stripe_start], hit)
            return sel, hit, gathered_k, gathered_v, k_cache, v_cache

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            k: pl.Tensor[[token_rows, HEAD_DIM], storage],
            v: pl.Tensor[[token_rows, HEAD_DIM], storage],
            slot_mapping: pl.Tensor[[1, token_rows], pl.INT32],
            slotidx: pl.Tensor[[1, cache_rows], pl.FP32],
            sel: pl.InOut[pl.Tensor[[token_rows, cache_rows], pl.FP32]],
            hit: pl.InOut[pl.Tensor[[1, cache_rows], pl.FP32]],
            gathered_k: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
            gathered_v: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
            k_cache: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
            v_cache: pl.InOut[pl.Tensor[[cache_rows, HEAD_DIM], storage]],
        ) -> tuple[
            pl.Tensor[[token_rows, cache_rows], pl.FP32],
            pl.Tensor[[1, cache_rows], pl.FP32],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
            pl.Tensor[[cache_rows, HEAD_DIM], storage],
        ]:
            with pl.spmd(owner_blocks):
                sel, hit, gathered_k, gathered_v, k_cache, v_cache = self.own_destinations(
                    k, v, slot_mapping, slotidx, sel, hit, gathered_k, gathered_v, k_cache, v_cache
                )
            return sel, hit, gathered_k, gathered_v, k_cache, v_cache

    return ReshapeAndCache


def build(shape):
    import torch
    from golden import TensorSpec

    token_rows = int(shape["M"])
    cache_rows = 4 * int(shape["N"])
    if token_rows <= 0 or token_rows % 64 or cache_rows % 128 or cache_rows <= token_rows:
        raise ValueError("Expected aligned positive token/cache extents and distinct-slot capacity")
    dtype = shape.get("dtype", "fp32")
    if dtype not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    storage = pl.BF16 if dtype == "bf16" else pl.FP32
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    specs = [
        TensorSpec("k", [token_rows, HEAD_DIM], torch_dtype, init_value=torch.randn),
        TensorSpec("v", [token_rows, HEAD_DIM], torch_dtype, init_value=torch.randn),
        TensorSpec(
            "slot_mapping", [1, token_rows], torch.int32,
            init_value=lambda: torch.randperm(cache_rows)[:token_rows].to(torch.int32).reshape(1, token_rows),
        ),
        TensorSpec(
            "slotidx", [1, cache_rows], torch.float32,
            init_value=lambda: torch.arange(cache_rows, dtype=torch.float32).reshape(1, cache_rows),
        ),
        TensorSpec("sel", [token_rows, cache_rows], torch.float32),
        TensorSpec("hit", [1, cache_rows], torch.float32),
        TensorSpec("gathered_k", [cache_rows, HEAD_DIM], torch_dtype),
        TensorSpec("gathered_v", [cache_rows, HEAD_DIM], torch_dtype),
        TensorSpec("k_cache", [cache_rows, HEAD_DIM], torch_dtype, init_value=torch.randn),
        TensorSpec("v_cache", [cache_rows, HEAD_DIM], torch_dtype, init_value=torch.randn),
    ]

    def golden_fn(tensors):
        slots = tensors["slot_mapping"].reshape(-1).long()
        tensors["sel"].zero_()
        tensors["sel"][torch.arange(token_rows), slots] = 1.0
        tensors["hit"][:] = tensors["sel"].sum(dim=0, keepdim=True)
        tensors["gathered_k"].zero_()
        tensors["gathered_v"].zero_()
        tensors["gathered_k"][slots] = tensors["k"]
        tensors["gathered_v"][slots] = tensors["v"]
        tensors["k_cache"][slots] = tensors["k"]
        tensors["v_cache"][slots] = tensors["v"]

    return make_program(token_rows, cache_rows, storage), specs, golden_fn, 1e-4, 1e-4

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
