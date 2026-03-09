# AGENTS.md - Agent Coding Guidelines for unify-easy-llm

## Project Overview

unify-easy-llm (ULM) is a one-click LLM training tool based on Firefly, supporting Nvidia GPU, Ascend NPU, and common models. Supports full-parameter fine-tuning, LoRA, and QLoRA.

## Build and Test Commands

```bash
# Full-parameter fine-tuning (GPU)
python train_unify.py --train_args_file train_args/sft-config-gpu.json

# Full-parameter fine-tuning (CPU)
python train_unify.py --train_args_file train_args/sft-config-cpu.json

# LoRA fine-tuning (GPU with DeepSpeed)
deepspeed --num_gpus=1 train_unify.py --train_args_file train_args/lora-config-gpu.json

# LoRA fine-tuning (NPU)
bash scripts/local_run_unify_lora_npu.sh
```

### Dependencies

```bash
pip install -r requirements-gpu-py310-torch210.txt  # GPU
pip install -r requirements-npu-py39-torch210.txt    # NPU
pip install pytest pyyaml setuptools loguru transformers peft deepspeed
```

### Testing

**No formal test suite exists yet.** Add tests using pytest when possible.

```bash
pytest                    # Run all tests
pytest tests/test_file.py # Run single file
pytest -k "pattern"      # Run matching tests
```

## Code Style Guidelines

### Imports (in order)

```python
# Standard library
import os
import sys
import torch
from dataclasses import dataclass, field

# Third-party
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Local
from component.collator import SFTDataCollator
from component.dataset import UnifiedSFTDataset
```

### Type Annotations

- Use `Optional[X]` instead of `X | None` for Python 3.9 compatibility
- Use dataclasses with `field` for configuration

```python
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class UnifyArguments:
    model_name_or_path: str = field(default="", metadata={"help": "Model path"})
    lora_rank: Optional[int] = field(default=64)
```

### Naming Conventions

- **Classes:** PascalCase (`UnifyArguments`, `SFTDataCollator`)
- **Functions/variables:** snake_case (`load_model`, `train_dataset`)
- **Constants:** UPPER_SNAKE_CASE
- **Private methods:** prefix with underscore

### Error Handling

```python
try:
    trainer, tokenizer, model = init_components(args, training_args)
    train_result = trainer.train()
except Exception as e:
    errMsg = f"模型训练异常，详细信息: {e}"
    logger.info(errMsg)
    traceback.print_exc()
    sys.exit(11)
```

### Directory Structure

```
unify-easy-llm/
├── train_unify.py       # Main entry point
├── component/
│   ├── argument.py     # Config dataclasses
│   ├── collator.py     # Data collation
│   ├── dataset.py      # Dataset classes
│   ├── trainer.py      # Training logic
│   ├── template.py     # Prompt templates
│   ├── eval.py         # Evaluation
│   ├── imports.py      # Dependency checks
│   └── train_utils.py  # Training utilities
├── train_args/         # JSON configs
├── scripts/            # Shell scripts
└── tools/              # Utilities
```

### Logging

Use `loguru.logger` for all logging:

```python
from loguru import logger

logger.info("开始训练。。。")
logger.info(f"Total model params: %.2fM" % (total / 1e6))
```

### Hardware Detection

```python
from component.imports import is_cuda_available, is_npu_available

if is_cuda_available():
    # GPU code
elif is_npu_available():
    # NPU code
```

### Model Loading

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### Distributed Training

```python
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
```

## Key Dependencies

`torch`, `transformers`, `peft`, `deepspeed`, `loguru`, `bitsandbytes` (optional)

## Common Issues

1. **Meta tensor error:** Use `empty_init=False` for ChatGLM
2. **Memory issues:** Use gradient checkpointing or quantization
3. **Model dtype:** Cast to fp16 for stability
