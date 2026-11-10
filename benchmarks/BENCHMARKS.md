# Benchmarks — PyPTO-Ops

All numbers: Ascend 910C (`a2a3`), driver 26.0.rc1, median effective latency,
`PYPTO_BENCH=1`, 100 rounds / 5 warmup, idle dies only.

## Method

Each operator's baseline is an **unfused decomposition written in the same DSL** — the
multi-pass, materialize-everything form of the same math. The holdout gain therefore
measures the value of fusion/scheduling itself, not a strawman in another toolchain.
Where a native aclnn equivalent exists, we additionally measured it on the same card
(end-to-end wall clock on both sides); those comparisons are listed in the README
catalog and the raw data is in `benchmarks/evidence/<op>/vs_aclnn.json` where present.

## Gates

An operator ships only after its final build passes a **sealed holdout**
(`contamination_flag: false`) with two gates:

1. accuracy against a literal torch golden (per-dtype tolerances; several quantization
   and data-movement ops are bit-exact, `rtol=atol=0`);
2. a minimum-latency gate on a fast shape (so a slow-but-correct regression cannot
   sneak in through the accuracy gate).

Candidates that are faster but wrong, or that fail the latency gate, are recorded as
FAIL in the evidence and never shipped.

## Evidence layout

```
benchmarks/evidence/<op>/
  <eid>.holdout.json     # sealed holdout event (verdict, latency, gain, flags)
  <eid>.record.json      # decision-level measurement record
  leaderboard-final.md   # leaderboard snapshot at domain close-out
  vs_aclnn.json          # where a native comparison exists
```

## Notes and honest caveats

- Three operators currently post sub-1.0× against their aclnn equivalent
  (`fused_rms_norm_gated` 0.62×, `gelu` 0.81×, `softmax_backward` 0.87×): included
  because they are correct and substantially faster than the unfused decomposition;
  the gap to native is on the record and under active optimization.
- `grouped_topk`'s absolute latency (1853 µs) reflects the routing-table shape of the
  evaluation preset; it is +84% against its own unfused decomposition.
- Numbers move: shipped kernels are refreshed as ongoing optimization converges. The catalog in the
  README always reflects the latest shipped kernels.
