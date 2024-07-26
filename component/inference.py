from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu


local_rank = 0
device = torch.device("npu", local_rank)
# device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/models/Qwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen1.5-7B-Chat",
                                           trust_remote_code=True)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)


