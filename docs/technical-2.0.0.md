# 技术设计文档 v2.0.0

## 版本信息
- **版本号**: 2.0.0
- **需求名称**: 支持Qwen3模型训练
- **创建日期**: 2026-03-09

## 1. 设计目标

本设计遵循最小侵入性原则，在不修改现有代码结构的前提下，通过扩展配置文件和注册机制添加Qwen3模型支持。

## 2. 技术方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    train_unify.py                        │
├─────────────────────────────────────────────────────────┤
│  component/template.py    │  添加qwen3模板注册           │
│  component/train_lora_utils.py  │  添加qwen3 target_modules  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 变更点清单

| 序号 | 文件 | 变更类型 | 描述 |
|-----|------|---------|------|
| 1 | component/template.py | 扩展 | 添加qwen3模板注册 |
| 2 | component/train_lora_utils.py | 扩展 | 添加qwen3的target_modules |
| 3 | train_args/qwen3-lora-config.json | 新增 | Qwen3 LoRA训练配置 |
| 4 | train_args/qwen3-sft-config.json | 新增 | Qwen3全参数训练配置 |
| 5 | tests/test_qwen3.py | 新增 | 单元测试 |

## 3. 详细设计

### 3.1 模板注册 (component/template.py)

参考Qwen2模板，添加Qwen3模板：

```python
register_template(
    template_name='qwen3',
    start_word=None,
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n',
    assistant_prompt_prefix='<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="你是一个有用的助手。"
)
```

### 3.2 LoRA Target Modules (component/train_lora_utils.py)

在target_modules_dict中添加：

```python
target_modules_dict = {
    # ... existing ...
    "qwen3": ['q_proj', 'v_proj'],
}
```

### 3.3 Tokenizer处理

Qwen3使用Qwen2Tokenizer，需要特殊处理：
- pad_token_id设置为eos_token_id
- bos_token_id设置为eos_token_id
- eos_token_id保持不变

### 3.4 配置文件示例

#### Qwen3 LoRA配置 (train_args/qwen3-lora-config.json)
```json
{
    "sft_type": "lora",
    "output_dir": "/path/to/output",
    "model_name_or_path": "/path/to/qwen3-model",
    "train_file": "/path/to/train.json",
    "prompt_template_name": "qwen3",
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    ...
}
```

#### Qwen3全参数配置 (train_args/qwen3-sft-config.json)
```json
{
    "sft_type": "full",
    "output_dir": "/path/to/output",
    "model_name_or_path": "/path/to/qwen3-model",
    "train_file": "/path/to/train.json",
    "prompt_template_name": "qwen3",
    ...
}
```

## 4. 测试设计

### 4.1 单元测试用例

| 测试用例 | 验证内容 | 预期结果 |
|---------|---------|---------|
| test_qwen3_template_registered | qwen3模板是否正确注册 | template_dict包含qwen3 |
| test_qwen3_target_modules | qwen3 target_modules配置 | 包含q_proj, v_proj |
| test_qwen3_template_format | 模板格式正确性 | 符合Qwen3格式规范 |

## 5. 兼容性说明

- 本变更不修改任何现有模型的配置
- 现有模型训练流程保持不变
- 向后兼容所有现有功能

## 6. 实施步骤

1. 修改component/template.py，添加qwen3模板
2. 修改component/train_lora_utils.py，添加qwen3 target_modules
3. 创建train_args/qwen3-lora-config.json
4. 创建train_args/qwen3-sft-config.json
5. 创建tests/test_qwen3.py单元测试
6. 更新README.md文档
