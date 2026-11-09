# Operator Catalog

Semantic index of the 27 operators in `pypto_ops`, grouped by role in the inference stack.
Every operator ships with a correctness contract test and a measured benchmark record
under `benchmarks/evidence/<op>/`.

## Normalization (6)

| Operator | Purpose |
|---|---|
| `rmsnorm_forward` | RMSNorm forward, the dominant normalization in modern LLMs |
| `fused_add_rmsnorm` | Residual add + RMSNorm in one pass (vLLM fusion pattern) |
| `fused_rms_norm_gated` | Gated RMSNorm (Mamba/GLA-style gated normalization) |
| `layernorm_forward` / `layernorm_backward` | LayerNorm forward and gradient |
| `skip_layernorm` | Skip-connection + LayerNorm fusion |

## Activations (5)

| Operator | Purpose |
|---|---|
| `silu` | SiLU activation |
| `gelu` | GELU activation |
| `geglu` | GEGLU gated MLP activation |
| `swiglu` | SwiGLU gated MLP activation |
| `mish` | Mish activation |

## Quantization (3)

| Operator | Purpose |
|---|---|
| `per_token_quant` | Per-token INT8 quantization for W8A8 GEMM pipelines |
| `per_token_dequant` | Per-token dequantization counterpart |
| `swiglu_quant` | SwiGLU fused with per-token quantization |

## MoE routing (5)

| Operator | Purpose |
|---|---|
| `topk_softmax` | Router softmax + top-k expert selection |
| `grouped_topk` | Grouped top-k routing (DeepSeek-style expert groups) |
| `moe_align_block_size` | Expert-parallel block alignment for grouped GEMM |
| `moe_permute` / `moe_unpermute` | Token permutation primitives for MoE dispatch/combine |

## KV cache & positional (3)

| Operator | Purpose |
|---|---|
| `reshape_and_cache` | Paged KV-cache write path (vLLM serving critical path) |
| `rotary_embedding` | Rotary positional embedding |
| `qk_l2norm_gate` | QK L2 normalization + gating for attention pre/post-processing |

## Sampling (1)

| Operator | Purpose |
|---|---|
| `gumbel_sample` | Gumbel-max sampling from logits |

## Losses & sequence primitives (4)

| Operator | Purpose |
|---|---|
| `softmax_forward` / `softmax_backward` | Numerically stable softmax and its gradient |
| `cross_entropy` | Cross-entropy loss forward |
| `causal_conv1d` | Causal 1-D convolution (Mamba-style sequence models) |
