
from collections import defaultdict

import argparse
from loguru import logger
import os
from os.path import join
import torch
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer, AutoConfig, AutoModelForCausalLM
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from component.collator import SFTDataCollator
from component.argument import UnifyArguments
from component.trainer import Trainer, LoRATrainer
from component.dataset import UnifiedSFTDataset, UnifiedEvalDataset
from component.eval import model_eval, metric_desc, metric_title
from component.common import print_rank_0
from component.callback import save_progress, load_progress
from component.template import template_dict

from itertools import chain
from tqdm import tqdm
import shutil
import time
import traceback
import sys
import os


target_modules_dict = {
    "bloom": ['query_key_value'],
    "qwen": ['c_attn','c_proj'],
    "qwen2": ['q_proj','v_proj'],
    "baichuan": ['W_pack','o_proj'],
    "chatglm": ['query_key_value']
}


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/full/qwen-7b-sft-full.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((UnifyArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)

    temp_output_path = training_args.output_dir

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args, temp_output_path


def load_tokenizer(args, model_config):

    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model_config.model_type == 'llama' else True
    )

    tokenizer_info = f"tokenizer.pad_token_id: {tokenizer.pad_token_id}, tokenizer.bos_token_id: {tokenizer.bos_token_id}, tokenizer.eos_token_id: {tokenizer.eos_token_id} \n tokenizer.pad_token: {tokenizer.pad_token}, tokenizer.bos_token: {tokenizer.bos_token}, tokenizer.eos_token: {tokenizer.eos_token}"
    logger.info(tokenizer_info)

    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id


    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"

    return tokenizer


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    logger.info('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    # 统计可训练参数中，各种类型参数分布
    logger.info('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    print("linear module name: ", list(lora_module_names))
    return list(lora_module_names)


def save_metrics_to_disk(trainer, train_result, training_args, val_metric):
    # 保存训练指标
    metrics = train_result.metrics
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.log_metrics("eval", val_metric)
    trainer.save_metrics("eval", val_metric)
    trainer.save_state()

    train_progress = load_progress(training_args)
    metrics = train_progress["metrics"]
    summaryRecord = []
    train_progress["progress"] = 100

    for key, value in val_metric.items():
        record = {
            "name": metric_title.get(key, ""),
            "title": metric_title.get(key, ""),
            "description": metric_desc.get(key, ""),
            "is_percent": True,
            "value": value
        }
        summaryRecord.append(record)
    metrics["summaryRecord"] = summaryRecord
    train_progress["metrics"] = metrics
    save_progress(training_args, train_progress)


def merge_lora_to_base_model(args, training_args):
    model_name_or_path = args.model_name_or_path
    adapter_name_or_path = training_args.output_dir
    save_path = args.model_temp_merge_path

    print("----------------- merge model weight: ----------------- \n model_name_or_path: ", model_name_or_path, "adapter_name_or_path: ", adapter_name_or_path, "save_path: ", save_path)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path, safe_serialization=True)



