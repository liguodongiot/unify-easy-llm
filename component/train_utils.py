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


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/full/qwen-7b-sft-full.json', help="")
    parser.add_argument("--local_rank", type=int, help="")

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


