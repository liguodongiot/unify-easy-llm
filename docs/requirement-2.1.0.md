# 需求设计文档 v2.1.0

## 版本信息
- **版本号**: 2.1.0
- **需求名称**: 支持Qwen3-30B-A3B模型训练
- **创建日期**: 2026-03-09

## 1. 需求概述

本需求旨在为unify-easy-llm项目添加Qwen3-30B-A3B模型的支持。Qwen3-30B-A3B是阿里云发布的Qwen3系列中的MoE（混合专家）模型，具有约30B活跃参数和更大的总参数量。该模型支持全参数微调、LoRA和QLoRA三种训练方式。

## 2. 背景分析

### 2.1 Qwen3-30B-A3B模型特点

Qwen3-30B-A3B是一款MoE（Mixture of Experts）模型，具有以下特点：
- **活跃参数量**: 约30B参数
- **专家总数**: 128个专家，8个活跃专家
- **架构**: 基于Qwen3 MoE架构
- **上下文长度**: 支持更长上下文
- **版本**: 提供Base和Instruct两种版本

### 2.2 与Dense模型的区别

| 特性 | Qwen3 Dense | Qwen3-30B-A3B (MoE) |
|------|-------------|---------------------|
| 架构 | Dense | MoE |
| 活跃参数 | 全部参数 | ~30B |
| 总参数 | ~30B | ~600B+ |
| LoRA target | q_proj, v_proj | 需要覆盖更多层 |
| 显存需求 | 较高 | QLoRA推荐 |

## 3. 功能需求

### 3.1 核心功能需求

| 需求编号 | 需求描述 | 优先级 |
|---------|---------|--------|
| REQ-001 | 支持Qwen3-30B-A3B的全参数微调 | 高 |
| REQ-002 | 支持Qwen3-30B-A3B的LoRA微调 | 高 |
| REQ-003 | 支持Qwen3-30B-A3B的QLoRA微调（推荐） | 高 |
| REQ-004 | 正确配置MoE模型的target_modules | 高 |
| REQ-005 | 支持Qwen3-30B-A3B的模型合并 | 中 |

### 3.2 LoRA Target Modules配置

Qwen3-30B-A3B MoE模型需要更完整的target_modules配置：

```python
target_modules = [
    # Attention层
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MLP层 (MoE专用)
    "gate_proj", "up_proj", "down_proj"
]
```

### 3.3 非功能需求

| 需求编号 | 需求描述 | 优先级 |
|---------|---------|--------|
| NFR-001 | 代码侵入性最小化 | 高 |
| NFR-002 | 保持与现有Qwen3配置的兼容性 | 高 |
| NFR-003 | 添加MoE模型专属配置 | 中 |

## 4. 训练配置要求

### 4.1 推荐配置

#### QLoRA配置（推荐）
- **lora_rank**: 64
- **lora_alpha**: 128
- **lora_dropout**: 0.1
- **target_modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

#### LoRA配置
- **lora_rank**: 32
- **lora_alpha**: 64
- **lora_dropout**: 0.05

### 4.2 硬件要求

| 训练方式 | GPU要求 | 预估显存 |
|---------|--------|---------|
| QLoRA | 单卡24GB | ~20GB |
| LoRA | 8卡A100 | ~80GB |
| 全参数 | 8卡A100 | ~160GB+ |

## 5. 验收标准

### 5.1 功能验收
- [ ] Qwen3-30B-A3B能够成功进行QLoRA微调
- [ ] Qwen3-30B-A3B能够成功进行LoRA微调
- [ ] Qwen3-30B-A3B能够成功进行全参数微调
- [ ] MoE target_modules配置正确
- [ ] LoRA权重能够正确合并

### 5.2 测试验收
- [ ] MoE target_modules配置测试通过
- [ ] 模板注册测试通过

## 6. 风险评估

| 风险编号 | 风险描述 | 影响程度 | 应对措施 |
|---------|---------|---------|---------|
| RISK-001 | MoE模型显存需求大 | 高 | 推荐使用QLoRA |
| RISK-002 | 部分层可能不支持LoRA | 中 | 使用完整target_modules |
| RISK-003 | 模型加载时间长 | 低 | 添加预加载处理 |
