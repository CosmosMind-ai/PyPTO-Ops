# PyPTO-Ops

[English](README.md) | **中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> 本仓库属于 **CosmosMind PyPTO 算子系列**（面向昇腾），全部内核由 [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1) 驱动。

面向**昇腾 910C 大模型推理**的 [FlagGems](https://github.com/FlagOpen/FlagGems)
式算子库，基于 [PyPTO](https://github.com/hw-native-sys/pypto) 编写——
覆盖 vLLM 级 serving 栈的融合与关键路径算子：归一化、激活/GLU、量化、
MoE 路由、KV-cache、采样与损失算子。

每个算子都须通过密封的精度+延迟 holdout 后方可发布。原始测量记录按算子发布于
`benchmarks/evidence/<op>/`。

## 状态

v0.1.0 —— **27 个算子**，全部在昇腾 910C 物理卡上验证。库在持续扩充，
新算子通过 holdout 后即合入。

## 算子目录

基线为同 DSL 编写的未融合分解（衡量融合本身的收益）；存在原生 aclnn 等价实现的，
并列给出同卡对比。完整协议见 `benchmarks/BENCHMARKS.md`，算子语义索引见
`docs/CATALOG.md`。

### 归一化

| 算子 | Holdout 结果 | vs 原生（aclnn） |
|---|---|---|
| `rmsnorm_forward` | 57.37 µs, +97.1% | — |
| `layernorm_forward` | 86.26 µs, +94.7% | — |
| `fused_add_rmsnorm` | 84.48 µs, +97.7% | **1.41×** vs aclnnAddRmsNorm |
| `fused_rms_norm_gated` | 83.91 µs, +72.4% | 0.62× vs aclnn 分解 |
| `skip_layernorm` | +98.0% | — |
| `layernorm_backward` | 29.58 µs, +89.1% | **2.45×** vs aclnn LN+LNBackward |
| `qk_l2norm_gate` | +85.2% | — |

### 激活 / GLU 族

| 算子 | Holdout 结果 | vs 原生（aclnn） |
|---|---|---|
| `silu` | +98.1% | — |
| `gelu`（tanh 近似） | 68.64 µs, +98.0% | 0.81× vs aclnnGeluV2 |
| `geglu` | +98.4% | — |
| `swiglu` | 71.47 µs, +97.7% | 1.13× vs aclnn Silu+Mul |
| `mish` | 44.59 µs, +98.4% | — |
| `swiglu_quant`（融合激活 + INT8 量化） | 50.99 µs, +76.0% | — |

### 量化

| 算子 | Holdout 结果 | vs 原生（aclnn） |
|---|---|---|
| `per_token_quant`（INT8，精确 RNE） | 33.88 µs, +77.5% | **3.81×** vs aclnn 9 算子分解 |
| `per_token_dequant` | 18.64 µs, +86.0% | — |

进行中：FP8 E4M3 per-token 量化（位精确编码器）、group-wise INT8 量化、
per-channel INT8 量化。

### MoE 路由

| 算子 | Holdout 结果 |
|---|---|
| `moe_permute`（dispatch） | 75.23 µs, +76.6%（位精确） |
| `moe_unpermute` | +98.0% |
| `moe_align_block_size` | 135.88 µs, +35.5% |
| `grouped_topk`（DeepSeek 式分组路由） | 1853.16 µs, +84.1% |
| `topk_softmax` | +98.7% |

### KV cache / 位置编码 / 采样 / 损失

| 算子 | Holdout 结果 | vs 原生（aclnn） |
|---|---|---|
| `reshape_and_cache`（分页 KV 写入） | +99.9% | — |
| `rotary_embedding` | +96.3% | — |
| `gumbel_sample` | +93.8% | — |
| `cross_entropy`（前向+反向） | 65.63 µs, +98.3% | **2.85×** vs aclnn CrossEntropyLoss+Grad |
| `softmax_forward` | 66.40 µs, +97.4% | — |
| `softmax_backward` | 94.0 µs, +96.9% | 0.87× vs aclnnSoftmaxBackward |
| `causal_conv1d`（短卷积，线性注意力配套） | 93.62 µs, +94.8% | — |

## 安装

```bash
pip install -e .
```

环境要求：昇腾 910C（`a2a3`），CANN ≥ 9.0，`pypto`、`pypto-lib`（其 `golden`
包支撑经验证的启动路径）、`torch`。

## 快速上手

```python
import torch
import pypto_ops

print(pypto_ops.available_ops())

mod = pypto_ops.get("fused_add_rmsnorm")
fn, specs, golden_fn, rtol, atol = mod.build(mod.DEFAULT_SHAPE)
# 经 golden harness 启动（见 tests/test_all.py），或将内核模块嵌入你自己的
# 运行时——每个内核都遵循 pypto-lib examples/intermediate 约定
#（build(shape) 入口 + __main__ CLI）。
```

## 测试

```bash
python tests/test_all.py --device 0            # 全目录 vs torch golden
python tests/test_all.py --device 0 --op per_token_quant
```

## 路线图

- [ ] FP8 / group / per-channel 量化族（进行中）
- [ ] MoE combine（`moe_sum`）与 log-softmax 反向
- [ ] 已覆盖算子的 vLLM-Ascend 直接替换路径
- [ ] 端到端 serving 收益（替换 vLLM-Ascend 内核路径前后的吞吐/时延）
- [ ] 新算子通过 holdout 后持续合入

## 相关仓库

- [PyPTO-FlashKDA](https://github.com/CosmosMind-ai/PyPTO-FlashKDA)：Kimi Delta Attention（前向 + 反向）
- [PyPTO-FlashDSA](https://github.com/CosmosMind-ai/PyPTO-FlashDSA)：昇腾 910C 上的 DeepSeek Sparse Attention（V4.1）

## 许可证

MIT，详见 `LICENSE`。
