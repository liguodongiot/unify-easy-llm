from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer import Trainer
from transformers import TrainingArguments

from component.common import print_rank_0
from component.argument import UnifyArguments

import os
import json

import torch
from loguru import logger
import dataclasses
from dataclasses import dataclass
import math



def save_progress_to_json(train_progress, json_path: str):
    """Save the content of this instance in JSON format inside `json_path`."""
    json_string = json.dumps(train_progress, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_string)


def save_progress(args, train_progress):
    json_path = args.output_dir + f"/progress.json"
    print("save path: ", json_path)
    save_progress_to_json(train_progress, json_path)


def load_progress(args):
    json_path = args.output_dir + f"/progress.json"
    print("load path: ", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(text)



class CustomCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, training_args: TrainingArguments, custom_arg:UnifyArguments):
        self.train_progress = load_progress(training_args)
        self.custom_arg = custom_arg
        print_rank_0("init callback ...")
        print_rank_0(custom_arg)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print_rank_0("begin train callback ...")
        self.current_step = 0
        self.current_epoch = 0

        self.train_progress["progress"] = 0
        save_progress(args, self.train_progress)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("end train callback ...")
        """
        "log_history": [
            {
            "epoch": 0.06,
            "learning_rate": 0,
            "loss": 6.7119,
            "step": 1
            },
            ...
            {
            "epoch": 0.95,
            "learning_rate": 0,
            "loss": 6.6951,
            "step": 15
            },
            {
            "epoch": 0.95,
            "step": 15,
            "total_flos": 6960821619916800.0,
            "train_loss": 6.784228515625,
            "train_runtime": 37.3538,
            "train_samples_per_second": 26.771,
            "train_steps_per_second": 0.402
            }
        ]
        """
        log_history = state.log_history
        loss_value = []
        for i, log in enumerate(log_history):
            print_rank_0(log)
            loss_v = log.get("loss")
            if loss_v is None:
                loss_v = log.get("train_loss")
            loss_value.append([log["step"], loss_v])
            if log["step"] >= state.max_steps:
                break

        trainRecord = []
        loss = {
            "name": "loss",
            "title": "Train Loss",
            "description": "训练集每个step的损失",
            "x": "step",
            "y": "Train Loss",
            "value": loss_value,
            "plot_type": "scatter"
        }
        trainRecord.append(loss)
        metrics = {}
        metrics["trainRecord"] = trainRecord
        self.train_progress["metrics"] = metrics
        print_rank_0(state)
        if state.is_local_process_zero:
            logger.info(f"current_epoch: {self.current_epoch}, current_step: {self.current_step}, "
                  f"max_steps: {state.max_steps}, global_step: {state.global_step}, ")
            # self.save_offical_progress(args, state, state.global_step - 1)
            logger.info("end train ...")


        self.train_progress["progress"] = 99
        save_progress(args, self.train_progress)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_epoch += 1

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_step += 1
        if state.is_local_process_zero:
            logger.info(f"step start, current_epoch: {self.current_epoch}, current_step: {self.current_step}, "
                  f"max_steps: {state.max_steps}, global_step: {state.global_step}, ")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.is_local_process_zero:
            logger.info(f"step end, current_epoch: {self.current_epoch}, current_step: {self.current_step}, "
                  f"max_steps: {state.max_steps}, global_step: {state.global_step}, ")
            # self.save_offical_progress(args, state, state.global_step - 1 - 1)

            if state.global_step % 5 == 0:
                self.train_progress["progress"] = math.floor(state.global_step*100/state.max_steps)
                save_progress(args, self.train_progress)
