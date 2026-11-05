"""PyPTO-Ops — operator library for LLM serving on Ascend 910C (PyPTO).

Lazy registry: `pypto_ops.get("rmsnorm_forward")` returns the kernel module
(with the pypto-lib example-style `build(shape)` entry and a `__main__` CLI).
"""
import importlib

_REGISTRY = {
    # normalization
    "rmsnorm_forward", "layernorm_forward", "fused_add_rmsnorm",
    "fused_rms_norm_gated", "skip_layernorm", "layernorm_backward",
    "qk_l2norm_gate",
    # activations / GLU
    "silu", "gelu", "geglu", "swiglu", "mish", "swiglu_quant",
    # quantization
    "per_token_quant", "per_token_dequant",
    # MoE routing
    "moe_permute", "moe_unpermute", "moe_align_block_size",
    "grouped_topk", "topk_softmax",
    # KV cache & position
    "reshape_and_cache", "rotary_embedding",
    # sampling
    "gumbel_sample",
    # loss & softmax
    "cross_entropy", "softmax_forward", "softmax_backward",
    # sequence mixing
    "causal_conv1d",
}

__version__ = "0.1.0"


def available_ops():
    """Names of all operators shipped in this release."""
    return sorted(_REGISTRY)


def get(name):
    """Import and return the kernel module for `name` (see available_ops())."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown op {name!r}; see available_ops()")
    return importlib.import_module(f"pypto_ops.kernels.{name}")
