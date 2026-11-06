# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SiLU with a fixed slab geometry and a selectable input-load cache policy.

Closed-form geometry: blk=8192, n_blocks=40, stage=4, buffers=1 for every
positive (M,N). TOTAL=M*N; NBCH=ceil(TOTAL/blk); K=ceil(NBCH/n_blocks).
Only chunk counts and tail bounds depend on shape. Chunk ids are clamped;
valid_shape loads and bounded stores cover partial final bursts.
"""
from golden import TensorSpec

import torch
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024}


RTOL = 1.6e-2
ATOL = 1.6e-2

DMA = 8192
NB = 40
STAGE = 4
BUFFERS = 1
VEC_BYTES_PER_ELEMENT = 16


def build(shape):
    M, N = int(shape["M"]), int(shape["N"])
    if M <= 0 or N <= 0:
        raise ValueError("SiLU needs positive extents, got %r" % (shape,))
    TOT = M * N
    NBCH = (TOT + DMA - 1) // DMA
    K = (NBCH + NB - 1) // NB
    print("slab_cache_policy geometry", {"M": M, "N": N, "tot": TOT,
          "dma_blk": DMA, "n_blocks": NB, "stage": STAGE, "buffers": BUFFERS,
          "vec_bytes_per_element": VEC_BYTES_PER_ELEMENT,
          "bursts": NBCH, "per_block": K, "last_chunk_elems": TOT - (NBCH - 1) * DMA})

    @pl.jit
    def silu_cache_policy(
        x: pl.Tensor[[M, N], pl.BF16],
        y: pl.Out[pl.Tensor[[M, N], pl.BF16]],
    ):
        xr = pl.reshape(x, [1, TOT])
        yr = pl.reshape(y, [1, TOT])
        for b in pl.spmd(NB, name_hint="silu_cache_policy"):
            for t in pl.pipeline(K, stage=STAGE):
                i = pl.min(b * K + t, NBCH - 1)
                o0 = i * DMA
                v = pl.min(DMA, TOT - o0)
                xt = pl.load(xr, [0, o0], [1, DMA], valid_shape=[1, v], cache=pl.CachePolicy.BYPASS)
                xf = pl.cast(xt, target_type=pl.FP32)
                sig = pl.add(pl.exp(pl.neg(xf)), 1.0)
                result = pl.cast(pl.div(xf, sig), target_type=pl.BF16)
                pl.store(result, [0, o0], yr)
        return y

    def specs():
        return [
            TensorSpec("x", [M, N], torch.bfloat16, init_value=torch.randn),
            TensorSpec("y", [M, N], torch.bfloat16),
        ]

    def golden(t):
        t["y"][:] = torch.nn.functional.silu(t["x"].float()).to(torch.bfloat16)

    return silu_cache_policy, specs(), golden, RTOL, ATOL

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
