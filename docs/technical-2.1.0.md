# 技术设计文档 v2.1.0

## 版本信息
- **版本号**: 2.1.0
- **需求名称**: 支持Qwen3-30B-A3B模型训练
- **创建日期**: 2026-03-09

## 1. 设计目标

本设计遵循最小侵入性原则，通过扩展现有配置和注册机制添加Qwen3-30B-A3B MoE模型支持。与普通Qwen3模型不同，MoE模型需要配置更多的LoRA target_modules。

## 2. 技术方案

### 2.1 变更点清单

| 序号 | 文件 | 变更类型 | 描述 |
|-----|------|---------|------|
| 1 | component/train_lora_utils.py | 扩展 | 添加qwen3MoE的target_modules |
| 2 | train_args/qwen3MoE-qlora-config.json | 新增 | Qwen3-30B-A3B QLoRA配置 |
| 3 | train_args/qwen3MoE-lora-config.json | 新增 | Qwen3-30B-A3B LoRA配置 |
| 4 | train_args/qwen3MoE-sft-config.json | 新增 | Qwen3-30B-A3B全参数配置 |
| 5 | tests/test_qwen3moe.py | 新增 | 单元测试 |

### 2.2 模板处理

Qwen3-30B-A3B使用与Qwen3相同的模板，无需新增模板注册。直接复用现有的qwen3模板。

## 3. 详细设计

### 3.1 LoRA Target Modules配置 (component/train_lora_utils.py)

添加qwen3MoE的target_modules，与普通Qwen3不同，需要包含MLP层：

```python
target_modules_dict = {
    # ... existing ...
    "qwen3MoE": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
}
```

### 3.2 配置文件设计

#### QLoRA配置 (推荐)
```json
{
    "sft_type": "qlora",
    "output_dir": "/path/to/output",
    "model_name_or_path": "/path/to/qwen3-30b-a3b",
    "train_file": "/path/to/train.json",
    "prompt_template_name": "qwen3",
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.1,
    "gradient_checkpointing": true,
    "fp16": true,
    "bits": 4
}
```

#### LoRA配置
```json
{
    "sft_type": "lora",
    "output_dir": "/path/to/output",
    "model_name_or_path": "/path/to/qwen3-30b-a3b",
    "train_file": "/path/to/train.json",
    "prompt_template_name": "qwen3",
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05
}
```

#### 全参数配置
```json
{
    "sft_type": "full",
    "output_dir": "/path/to/output",
    "model_name_or_path": "/path/to/qwen3-30b-a3b",
    "train_file": "/path/to/train.json",
    "prompt_template_name": "qwen3"
}
```

### 3.3 QLoRA特殊处理

根据PEFT最佳实践，QLoRA训练Qwen3-30B-A3B时需要：
1. 使用4-bit NF4量化
2. 使用double quantization
3. 启用gradient checkpointing
4. LoRA rank建议使用64以获得更好的质量

## 4. 测试设计

### 4.1 单元测试用例

| 测试用例 | 验证内容 | 预期结果 |
|---------|---------|---------|
| test_qwen3moe_target_modules_exists | qwen3MoE配置存在 | 在target_modules_dict中 |
| test_qwen3moe_target_modules_complete | 完整attention+MLP | 包含7个模块 |
| test_qwen3moe_config_files_exist | 配置文件存在 | 3个配置文件都存在 |

## 5. 兼容性说明

- 本变更复用现有的qwen3模板
- 不影响现有任何模型的配置
- 向后兼容所有现有功能

## 6. 实施步骤

1. 修改component/train_lora_utils.py，添加qwen3MoE target_modules
2. 创建train_args/qwen3MoE-qlora-config.json
3. 创建train_args/qwen3MoE-lora-config.json
4. 创建train_args/qwen3MoE-sft-config.json
5. 创建tests/test_qwen3moe.py单元测试
6. 更新README.md文档
