from loguru import logger
import os
from os.path import join
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from component.collator import SFTDataCollator
from component.trainer import Trainer, LoRATrainer
from component.dataset import UnifiedSFTDataset, UnifiedEvalDataset
from component.eval import model_eval, metric_desc, metric_title
from component.common import print_rank_0
from component.callback import CustomCallback, save_progress, load_progress
from component.template import template_dict
from component.train_utils import setup_everything, load_tokenizer, save_metrics_to_disk
from component.train_lora_utils import verify_model_dtype, find_all_linear_names, target_modules_dict, merge_lora_to_base_model
from component.imports import is_bnb_available


import traceback
import sys
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def load_model(args, training_args, model_config):
    """
    加载模型
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    training_args.ddp_find_unused_parameters = False if ddp else None

    # 初始化model
    if model_config.model_type == 'chatglm': 
        # 修复: Cannot copy out of meta tensor; no data!
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            empty_init= False,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))    

    return model

def load_lora_model(args, training_args, model_config):
    """
    加载模型
    """
    training_args.ddp_find_unused_parameters = False

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    # device_map = {'': local_rank}
    # device_map = "auto"
    import torch.distributed as dist
    dist_initialized = dist.is_initialized()
    if not dist_initialized:
        logger.info("----auto")
        device_map = "auto"
    else:
        logger.info("----ddp")
        device_map = {'': local_rank}

    logger.info(f'微调类型：{args.sft_type}')
    use_cache = False if training_args.gradient_checkpointing else True
    if args.sft_type == "lora":
        load_in_8bit = bool(args.load_in_8bit)
        logger.info(f'是否8bit加载：{load_in_8bit}')
        if not load_in_8bit:
            quant_config = None
        else:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.sft_type == "qlora":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
    else:
        logger.info(f'目前暂不支持该微调类型：{args.sft_type}')
        sys.exit(12)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        use_cache = use_cache,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quant_config
    )
    
    if (not load_in_8bit) and args.sft_type != 'qlora':
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    else:
        # casts all the non int8 modules to full precision (fp32) for stability
        logger.info(f"gradient_checkpointing: {training_args.gradient_checkpointing}")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    logger.info(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')

    # 找到所有需要插入adapter的全连接层
    #target_modules = find_all_linear_names(model)
    # target_modules = ['query_key_value']
    # target_modules = ['q_proj','v_proj']
    target_modules = target_modules_dict.get(model_config.model_type)
    # 初始化lora配置
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

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    return model




def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    assert args.task_type in ['sft'], 'task_type should be in [sft]'
    # 初始化模型及Tokenizer
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = load_tokenizer(args, model_config)
    
    
    # 初始化dataset和collator
    prompt_template = template_dict[args.prompt_template_name]
    train_dataset = UnifiedSFTDataset(args.train_file, tokenizer, args.max_seq_length, prompt_template)
    # 加载collator
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    # 回调
    tianqiong = CustomCallback(training_args, args)

    if args.sft_type == 'lora' or args.sft_type == 'qlora':
        model = load_lora_model(args, training_args, model_config)
        # 初始化Trainer
        trainer = LoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_loss=loss_func
            callbacks=[tianqiong]
        )
    else:
        model = load_model(args, training_args, model_config)
        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_loss=loss_func
            callbacks=[tianqiong]
        )

    return trainer, tokenizer, model



def main():
    # 进行一些配置和检查
    args, training_args, temp_output_path = setup_everything()
    #time.sleep(30)
    print("training_args.output_dir: ", training_args.output_dir, "temp_output_path: ", temp_output_path)

    try:
        # 加载各种组件
        trainer, tokenizer, model = init_components(args, training_args)
        # 开始训练
        logger.info("开始训练。。。")
        train_result = trainer.train()
        logger.info(f"保存模型权重, path: {training_args.output_dir}")
        if args.sft_type == 'lora' or args.sft_type == 'qlora':
            trainer.save_model(training_args.output_dir)  # Saves the tokenizer too
            merge_lora_to_base_model(args, training_args)
        else:
            trainer.save_model(training_args.output_dir)  # Saves the tokenizer too
            tokenizer.save_pretrained(training_args.output_dir)

        logger.info("模型评估。。。")
        prompt_template = template_dict[args.prompt_template_name]
        val_dataset = UnifiedEvalDataset(args.train_file, tokenizer, args.max_seq_length, prompt_template)
        val_metric = model_eval(training_args, tokenizer, model, val_dataset)
        
        
        if trainer.is_world_process_zero():
            # 保存训练指标
            save_metrics_to_disk(trainer, train_result, training_args, val_metric)            
    except Exception as e:
        errMsg = f"模型训练异常，详细信息: {e}"
        logger.info(errMsg)
        traceback.print_exc()
        train_progress_json = load_progress(training_args)
        train_progress_json["errMsg"] = errMsg
        save_progress(training_args, train_progress_json)
        sys.exit(11)


if __name__ == "__main__":
    main()
