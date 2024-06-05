
# pip install rouge-chinese
# pip install rouge_chinese nltk jieba datasets
# 中文rouge评估

#import evaluate
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import jieba
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from component.common import print_rank_0
from component.imports import is_npu_available, is_cuda_available


metric_desc = {
    "rouge-1": "忽略停用词后，将模型生成的结果和标准结果按unigram拆分后，计算出的召回率。unigram指将句子或文本中的每个单词都单独作为一个基本单元，不考虑单词之间的顺序。",
    "rouge-2": "忽略停用词后，将模型生成的结果和标准结果按bigram拆分后，计算出的召回率。bigram指将句子或文本中的每个相邻的单词对都作为一个基本单元，用于描述两个单词之间的顺序关系。",
    "rouge-l": "忽略停用词后，衡量了模型生成的结果和标准结果的最长公共子序列，并计算出召回率。最长公共子序列指两个或多个字符串最长的子序列，这些子序列在每个字符串中都存在，且它们的顺序相同。",
    "bleu-4": "忽略停用词后，用于评估模型生成的句子和实际句子的差异的指标，值为unigram，bigram，trigram，4-grams的加权平均。"
}


metric_title = {
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2",
    "rouge-l": "ROUGE-L",
    "bleu-4": "BLEU-4"
}

# Metric
def compute_metrics(preds, labels):

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }

    for pred, label in zip(preds, labels):
        if pred is None or label is None or len(pred)==0:
            print_rank_0("存在空值---"+label+"\n---"+pred+"\n=====================")
            continue
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = round(float(np.mean(v)), 3)
    return score_dict


# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)
def model_eval(args, tokenizer, model, dataloader):
    local_rank = torch.distributed.get_rank()
    print("model eval start，local_rank：{}, model output path: {}".format(local_rank, args.output_dir))
    #model = AutoModelForCausalLM.from_pretrained(args.output_dir, trust_remote_code=True)
    model.eval()
    preds = []
    labels = []
    i = 0
    
    if is_npu_available(check_device=True):
        device = torch.device("npu", local_rank)
    elif is_cuda_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    # cuda = 'cuda:'+str(local_rank)
    for data in dataloader:
        i += 1
        text = data["text"]
        lable = data["label"]

        # print_rank_0("评估数据"+str(i)+"---"+text+"\n-----------------------")

        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        # inputs = inputs.cuda()
        inputs = inputs.to(device)
        model.to(device)
        # pred = model.generate(**inputs, max_length=args.max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
        pred = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.7, temperature=0.95)
        pred_label = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

        pred_label = pred_label[len(text):]
        # pred_label = tokenizer.decode(pred[0], skip_special_tokens=True)
        preds.append(pred_label)
        labels.append(lable)
        print_rank_0("==========\n评估数据"+str(i)+"\n---"+text+"\n---"+lable+"\n---"+pred_label+"\n=====================")

    score_dict = compute_metrics(preds, labels)
    print("model eval end...")
    return score_dict



