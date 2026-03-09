# 需求设计文档 v2.0.0

## 版本信息
- **版本号**: 2.0.0
- **需求名称**: 支持Qwen3模型训练
- **创建日期**: 2026-03-09

## 1. 需求概述

本需求旨在为unify-easy-llm项目添加Qwen3系列模型的支持，包括Qwen3-Base和Qwen3-Instruct变体。Qwen3是阿里云最新发布的大语言模型系列，支持全参数微调、LoRA和QLoRA三种训练方式。

## 2. 背景分析

### 2.1 现有模型支持情况
当前项目已支持以下模型：
- baichuan/baichuan2
- qwen/qwen1.5/qwen2
- glm3/glm4
- bloom
- llama

### 2.2 Qwen3模型特点
Qwen3具有以下特点：
- 采用了与Qwen2类似的架构设计
- 支持更长的上下文长度
- 提供了Base（预训练）和Instruct（指令微调）两种版本
- 参数量覆盖0.5B到72B范围
- 使用与Qwen2相似的tokenizer，但存在细微差异

## 3. 功能需求

### 3.1 核心功能需求

| 需求编号 | 需求描述 | 优先级 |
|---------|---------|--------|
| REQ-001 | 支持Qwen3-Base模型的全参数微调 | 高 |
| REQ-002 | 支持Qwen3-Instruct模型的微调 | 高 |
| REQ-003 | 支持Qwen3的LoRA微调方式 | 高 |
| REQ-004 | 支持Qwen3的QLoRA微调方式 | 高 |
| REQ-005 | 正确处理Qwen3特殊的tokenizer配置 | 高 |
| REQ-006 | 支持Qwen3模型的推理和合并 | 中 |

### 3.2 非功能需求

| 需求编号 | 需求描述 | 优先级 |
|---------|---------|--------|
| NFR-001 | 代码侵入性最小化，不影响现有模型 | 高 |
| NFR-002 | 保持与现有训练流程的兼容性 | 高 |
| NFR-003 | 添加相应的单元测试 | 中 |

## 4. 训练配置要求

### 4.1 模型配置
- **model_type**: qwen3
- **prompt_template_name**: qwen3
- **支持的训练类型**: full, lora, qlora

### 4.2 LoRA配置
- **target_modules**: q_proj, v_proj
- **lora_rank**: 默认64
- **lora_alpha**: 默认16
- **lora_dropout**: 默认0.05

### 4.3 Tokenizer配置
- 使用Qwen3专用的tokenizer
- 需要特殊处理pad_token、bos_token、eos_token
- 适用use_fast=True

## 5. 验收标准

### 5.1 功能验收
- [ ] Qwen3-Base模型能够成功进行全参数微调
- [ ] Qwen3-Instruct模型能够成功进行微调
- [ ] Qwen3的LoRA微调能够正常工作
- [ ] Qwen3的QLoRA微调能够正常工作
- [ ] LoRA权重能够正确合并到基础模型

### 5.2 测试验收
- [ ] 模板注册测试通过
- [ ] target_modules配置测试通过
- [ ] Tokenizer处理测试通过

## 6. 风险评估

| 风险编号 | 风险描述 | 影响程度 | 应对措施 |
|---------|---------|---------|---------|
| RISK-001 | Qwen3 tokenizer与Qwen2存在细微差异 | 中 | 在tokenizer加载时添加特殊处理逻辑 |
| RISK-002 | 某些Qwen3变体可能不支持QLoRA | 低 | 仅在模型支持时启用QLoRA |
