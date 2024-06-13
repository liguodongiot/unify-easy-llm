from torch.utils.data import Dataset
from loguru import logger
import json
import random
import math
import os

def load_local_dataset(file_path):
    logger.info('Loading data: {}'.format(file_path))
    file_names = os.listdir(file_path)
    filename_list = [os.path.join(file_path, file_name) for file_name in file_names if os.path.isfile(os.path.join(file_path, file_name))]
    logger.info('Loading dataset file list: {}'.format(filename_list))

    if filename_list is None or len(filename_list) == 0:
        logger.info("数据集文件列表为空，直接退出！！！")
        os._exit(1)
    
    data_list = []
    for file_name in  filename_list:

        data_temp_list = json.load(open(file_name, "r", encoding="utf-8"))
        data_list.extend(data_temp_list)
        logger.info("filename: {} , size: {} .".format(file_name, len(data_temp_list)))

    logger.info("there are {} data in dataset .".format(len(data_list)))
    return data_list


class UnifiedSFTDataset(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.start_word = template.start_word
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_prompt_prefix=template.assistant_prompt_prefix
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        logger.info("Tokenizer --- bos_token: {}, eos_token: {}, bos_token_id: {}, eos_token_id: {}".format(
            self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id))
        self.data_list = load_local_dataset(file)
        print(self.data_list[0])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {start_word}{system_format}{user_format}{assistant_prompt_prefix}{assistant_format}{user_format}{assistant_prompt_prefix}{assistant_format}...
        data = self.data_list[index]
        conversations = data['conversations']
        # prompt = data['prompt']

        input_ids = []
        target_mask = []
        
        if self.start_word is not None:
            start_word_ids = self.tokenizer.encode(self.start_word, add_special_tokens=False)
            input_ids += start_word_ids
            target_mask += [0] * len(start_word_ids)

        if self.system_format is not None:
            prompt = ""
            if data['conversations'][0].get("from") == "system":
                prompt = data['conversations'][0].get("value")
            system = prompt if prompt is not None and len(prompt) != 0 else self.system
            
            if system is not None:
                system_text = self.system_format.format(content=system)
                system_text_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                input_ids += system_text_ids
                target_mask += [0] * len(system_text_ids)

        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            value_str = conv['value'].strip()
            from_str = conv['from'].strip()

            if from_str == "assistant":
                if self.assistant_prompt_prefix is not None and len(self.assistant_prompt_prefix) != 0:
                    assistant_prompt_prefix_ids = self.tokenizer.encode(self.assistant_prompt_prefix, add_special_tokens=False)
                    input_ids += assistant_prompt_prefix_ids
                    target_mask += [0] * len(assistant_prompt_prefix_ids)
                outputs =  self.assistant_format.format(content=value_str)
                outputs_ids = self.tokenizer.encode(outputs, add_special_tokens=False)
                input_ids += outputs_ids
                target_mask += [1] * len(outputs_ids)
            elif from_str == "user":
                inputs = self.user_format.format(content=value_str)
                inputs_ids = self.tokenizer.encode(inputs, add_special_tokens=False)
                input_ids += inputs_ids
                target_mask += [0] * len(inputs_ids)

        assert len(input_ids) == len(target_mask)

        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        if index == 0:
            # text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            print(text)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class UnifiedEvalDataset(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer

        self.template_name = template.template_name
        self.start_word = template.start_word
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_prompt_prefix=template.assistant_prompt_prefix
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length

        logger.info("Tokenizer --- bos_token: {}, eos_token: {}, bos_token_id: {}, eos_token_id: {}".format(
            self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id))

        data_list = load_local_dataset(file)
        total_num = len(data_list)
        logger.info("there are {} data in dataset".format(total_num))
        if total_num < 100:
            self.data_list = data_list
        else:   
            eval_num = math.floor(total_num * 0.01)
            eval_num = min(100, eval_num)
            eval_list = random.sample(data_list, eval_num)
            self.data_list = eval_list
            # self.data_list = data_list
        print("there are {} eval data in dataset：".format(len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]

        conversations = data['conversations']
        # prompt = data['prompt']

        text = ""
        label = ""

        if self.start_word is not None:
            text += self.start_word

        if self.system_format is not None:
            prompt = ""
            if data['conversations'][0].get("from") == "system":
                prompt = data['conversations'][0].get("value")
            system = prompt if prompt is not None and len(prompt) != 0 else self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                text += system_text
        
        # 保证最后一句为助手的生成
        if conversations[len(conversations)-1]['from'].strip() == "user":
            conversations = conversations[:-1]

        for i, conv in enumerate(conversations):
            value_str = conv['value'].strip()
            from_str = conv['from'].strip()

            if from_str == "assistant":
                if self.assistant_prompt_prefix is not None and len(self.assistant_prompt_prefix) != 0:
                    text += self.assistant_prompt_prefix
                if len(conversations) - 1 == i:
                    label = self.assistant_format.format(content=value_str)
                else:
                    outputs = self.assistant_format.format(content=value_str)
                    text += outputs
            elif from_str == "user":
                inputs = self.user_format.format(content=value_str)
                text += inputs

        inputs = {
            "text": text,
            "label": label
        }
        return inputs



# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# if __name__ == "__main__":
#     train_file = "/home/guodong.li/workspace/data/dianxiao/new"
#     tokenizer = AutoTokenizer.from_pretrained(
#         "/home/guodong.li/workspace/model/baichuan2-7b",
#         trust_remote_code=True,
#         use_fast=True
#     )
#     max_seq_length = 1024
#     train_dataset = CustomDataset(train_file, tokenizer, max_seq_length)
#     total_token = 0
#     for data_set in train_dataset:
#         # print(data_set)
#         input_ids_token = len(data_set['input_ids'])
#         total_token += input_ids_token

#     print(total_token)


# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
 
# from template import template_dict

# if __name__ == "__main__":
#     # train_file = "/home/guodong.li/workspace/data/new"
#     train_file = "/home/guodong.li/workspace/data/dianxiao/new-test"
#     tokenizer = AutoTokenizer.from_pretrained(
#         "/home/guodong.li/workspace/model/bloom-2b6-zh",
#         trust_remote_code=True,
#         use_fast=True
#     )

#     # template = template_dict["qwen"]
#     # template = template_dict["chatglm3"]
#     # template = template_dict["baichuan"]

    
#     template = template_dict["bloom"]
#     max_seq_length = 1024

#     train_dataset = UnifiedSFTDataset(train_file, tokenizer, max_seq_length, template)
    
#     for data_set in train_dataset:
#         print(data_set['input_ids'])
#         print(data_set['attention_mask'])
#         print(data_set['target_mask'])
#         break
    
#     eval_dataset = UnifiedEvalDataset(train_file, tokenizer, max_seq_length, template)
    
#     print("------------------")

#     i = 0 
#     for eval_set in eval_dataset:
#         # if i ==10:
#         #     break
#         print(i, eval_set['label'])

#         i += 1
#         if eval_set['label'] is None or len(eval_set['label']) == 0:
#             print(i)
#             print(eval_set['text'])
#             print(eval_set['label'])
#     print("------------------")

