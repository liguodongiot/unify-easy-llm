from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PlatformArguments:
    model_metrics_path: str = field(default="", metadata={"help": "模型进度指标"})
    model_temp_output_path: str = field(default="/workspace/temp/outputs", metadata={"help": "模型临时输出路径"})
    model_temp_merge_path: str = field(default="/workspace/temp/merges", metadata={"help": "模型合并临时路径"})
    model_temp_merge_path: str = field(default="/workspace/temp/merges", metadata={"help": "模型合并临时路径"})
    prompt_template_name: str = field(default="default", metadata={"help": "提示模板名"})


@dataclass
class UnifyArguments(PlatformArguments):
    max_seq_length: int = field(default=0, metadata={"help": "输入最大长度"})
    train_file: str = field(default="", metadata={"help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    model_name_or_path: str = field(default="", metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[sft, pretrain]"})

    sft_type: str = field(default="", metadata={"help": "微调类型：[lora, qlora]"})
    load_in_8bit: bool = field(default=False, metadata={"help": "模型是否8bit加载"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
