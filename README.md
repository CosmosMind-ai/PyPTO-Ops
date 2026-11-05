# PyPTO-Ops

A [FlagGems](https://github.com/FlagOpen/FlagGems)-style operator library for
**LLM inference on Ascend 910C**, written in [PyPTO](https://github.com/hw-native-sys/pypto) —
fusion and critical-path operators covering the vLLM-class serving stack:
normalization, activations/GLU, quantization, MoE routing, KV-cache, sampling,
and loss kernels.

Every operator ships only after passing a sealed correctness-and-latency holdout.
Raw measurement records are published per operator under `benchmarks/evidence/<op>/`.

## Status

v0.1.0 — **27 operators**, all verified on physical Ascend 910C hardware. The library
is actively growing; new operators land here as they pass holdout.

## Operator catalog

Baselines are unfused decompositions written in the same DSL (measuring the value of
fusion itself); where a native aclnn equivalent exists, a same-card comparison is listed
alongside. Full protocol: `benchmarks/BENCHMARKS.md`. A semantic index of all operators:
`docs/CATALOG.md`.

### Normalization

| Operator | Holdout result | vs native (aclnn) |
|---|---|---|
| `rmsnorm_forward` | 57.37 µs, +97.1% | — |
| `layernorm_forward` | 86.26 µs, +94.7% | — |
| `fused_add_rmsnorm` | 84.48 µs, +97.7% | **1.41×** vs aclnnAddRmsNorm |
| `fused_rms_norm_gated` | 83.91 µs, +72.4% | 0.62× vs aclnn decomposition |
| `skip_layernorm` | +98.0% | — |
| `layernorm_backward` | 29.58 µs, +89.1% | **2.45×** vs aclnn LN+LNBackward |
| `qk_l2norm_gate` | +85.2% | — |

### Activations / GLU family

| Operator | Holdout result | vs native (aclnn) |
|---|---|---|
| `silu` | +98.1% | — |
| `gelu` (tanh approx) | 68.64 µs, +98.0% | 0.81× vs aclnnGeluV2 |
| `geglu` | +98.4% | — |
| `swiglu` | 71.47 µs, +97.7% | 1.13× vs aclnn Silu+Mul |
| `mish` | 44.59 µs, +98.4% | — |
| `swiglu_quant` (fused activation + INT8 quant) | 50.99 µs, +76.0% | — |

### Quantization

| Operator | Holdout result | vs native (aclnn) |
|---|---|---|
| `per_token_quant` (INT8, exact RNE) | 33.88 µs, +77.5% | **3.81×** vs aclnn 9-op decomposition |
| `per_token_dequant` | 18.64 µs, +86.0% | — |

In progress: FP8 E4M3 per-token quant (bit-exact encoder), group-wise INT8 quant,
per-channel INT8 quant.

### MoE routing

| Operator | Holdout result |
|---|---|
| `moe_permute` (dispatch) | 75.23 µs, +76.6% (bit-exact) |
| `moe_unpermute` | +98.0% |
| `moe_align_block_size` | 135.88 µs, +35.5% |
| `grouped_topk` (DeepSeek-style grouped routing) | 1853.16 µs, +84.1% |
| `topk_softmax` | +98.7% |

### KV cache / position / sampling / loss

| Operator | Holdout result | vs native (aclnn) |
|---|---|---|
| `reshape_and_cache` (paged KV write) | +99.9% | — |
| `rotary_embedding` | +96.3% | — |
| `gumbel_sample` | +93.8% | — |
| `cross_entropy` (fwd+bwd) | 65.63 µs, +98.3% | **2.85×** vs aclnn CrossEntropyLoss+Grad |
| `softmax_forward` | 66.40 µs, +97.4% | — |
| `softmax_backward` | 94.0 µs, +96.9% | 0.87× vs aclnnSoftmaxBackward |
| `causal_conv1d` (short conv, linear-attn companion) | 93.62 µs, +94.8% | — |

## Installation

```bash
pip install -e .
```

Requirements: Ascend 910C (`a2a3`), CANN ≥ 9.0, `pypto`, `pypto-lib` (its `golden`
package backs the validated launch path), `torch`.

## Quickstart

```python
import torch
import pypto_ops

print(pypto_ops.available_ops())

mod = pypto_ops.get("fused_add_rmsnorm")
fn, specs, golden_fn, rtol, atol = mod.build(mod.DEFAULT_SHAPE)
# launch via the golden harness (see tests/test_all.py), or embed the kernel
# module in your own runtime — each kernel follows the pypto-lib
# examples/intermediate conventions (build(shape) entry + __main__ CLI).
```

## Testing

```bash
python tests/test_all.py --device 0            # full catalog vs torch goldens
python tests/test_all.py --device 0 --op per_token_quant
```

## Roadmap

- [ ] FP8 / group / per-channel quantization family (in progress)
- [ ] MoE combine (`moe_sum`) and log-softmax backward
- [ ] vLLM-Ascend drop-in replacement path for the covered operators
- [ ] End-to-end serving deltas (replacing the vLLM-Ascend kernel path, before/after
      throughput and latency)
- [ ] New operators land continuously as they pass holdout

## Related repositories

- [PyPTO-FlashKDA](https://github.com/CosmosMind-ai/pypto-flashkda): Kimi Delta Attention (forward + backward)
- [PyPTO-FlashDSA](https://github.com/CosmosMind-ai/pypto-flashdsa): DeepSeek Sparse Attention (V4.1) on 910C

## License

MIT. See `LICENSE`.
