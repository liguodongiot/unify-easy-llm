from dataclasses import dataclass
from typing import Dict

@dataclass
class Template:
    template_name:str
    start_word: str
    system_format: str
    user_format: str
    assistant_prompt_prefix: str
    assistant_format: str
    system: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, start_word, 
    system_format, user_format, 
    assistant_prompt_prefix , assistant_format, system):
    template_dict[template_name] = Template(
        template_name=template_name,
        start_word=start_word,
        system_format=system_format,
        user_format=user_format,
        assistant_prompt_prefix=assistant_prompt_prefix,
        assistant_format=assistant_format,
        system=system
    )

# 注册template
register_template(
    template_name='default',
    start_word = None, 
    system_format='System: {content}\n',
    user_format='User: {content}\n',
    assistant_prompt_prefix='Assistant: ',
    assistant_format='{content}\n',
    system="你是一个有用的助手。"
)

register_template(
    template_name='qwen',
    start_word = None, 
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n',
    assistant_prompt_prefix='<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="你是一个有用的助手。"
)


register_template(
    template_name='qwen2',
    start_word = None, 
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n',
    assistant_prompt_prefix='<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="你是一个有用的助手。"
)


register_template(
    template_name='chatglm3',
    start_word = None, 
    system_format='<|sysmtem|>\n{content}',
    user_format='<|user|>\n{content}',
    assistant_prompt_prefix='<|assistant|>\n',
    assistant_format='{content}',
    system="你是一个有用的助手，请仔细遵循用户的指示进行响应。"
)


register_template(
    template_name='glm4',
    start_word = "[gMASK]<sop>", 
    system_format='<|sysmtem|>\n{content}',
    user_format='<|user|>\n{content}',
    assistant_prompt_prefix='<|assistant|>\n',
    assistant_format='{content}',
    system="你是一个有用的助手，请仔细遵循用户的指示进行响应。"
)

register_template(
    template_name='baichuan',
    start_word = None, 
    system_format='<s>{content}</s>',
    user_format='<reserved_102>{content}',
    assistant_prompt_prefix='<reserved_103>',
    assistant_format='{content}',
    system=None
)

register_template(
    template_name='baichuan2',
    start_word = None, 
    system_format='<s>{content}</s>',
    user_format='<reserved_106>{content}</s>',
    assistant_prompt_prefix='<reserved_107>',
    assistant_format='{content}</s>',
    system=None
)

register_template(
    template_name='bloom',
    start_word=None,
    system_format='{content}',
    user_format='{content}</s>',
    assistant_prompt_prefix=None,
    assistant_format='{content}</s>',
    system=None
)

register_template(
    template_name='base',
    start_word="<s>",
    system_format='{content}\n\n',
    user_format='{content}</s>',
    assistant_prompt_prefix=None,
    assistant_format='{content}</s>',
    system=None
)

