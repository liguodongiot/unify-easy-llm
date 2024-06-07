# unify-easy-llm（ULM）

本项目基于Firefly改造旨在打造一个简易的一键式大模型训练工具，支持Nvidia GPU、Ascend NPU等不同硬件以及常用的大模型。

## 微调方法

- 全量微调
- LoRA微调
- QLoRA微调

## 支持的模型

- baichuan2-7b/13b
- qwen-7b/14b
- qwen1.5-7b/14b

## 支持的硬件

- Nvidia GPU: A800、H800、RTX 3090、RTX 4090
- Ascend NPU: 910A、910B


## 环境安装

### Ascend NPU


1. 安装NPU驱动和固件。
2. 需预先安装 Ascend Docker Runtime。[软件包](https://gitee.com/ascend/ascend-docker-runtime/releases/tag/v6.0.0-RC1)、[安装教程](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/clusterscheduling/dockerruntimeug/dlruntime_ug_007.html)。

### Nvidia GPU

1. 安装GPU驱动。
2. 需预先安装 Docker。[安装教程](https://github.com/liguodongiot/llm-action/blob/main/docs/llm-base/a800-env-install.md#nvidia-docker-%E5%AE%89%E8%A3%85)


## 训练脚本

### 全参微调

Ascend NPU:
```
docker run -it -u root \
--network host \
--shm-size 4G \
-e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
-v /etc/localtime:/etc/localtime \
-v /var/log/npu/:/usr/slog \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /data/containerd/workspace/:/workspace \
harbor.llm.io/base/llm-train-unify:v1-20240603 \
/bin/bash -c '. ~/.bashrc &&  conda activate llm-dev && sh /workspace/llm-train/scipts/local_run_unify_sft_npu.sh'

```

### LoRA


Ascend NPU:

```
docker run -it -u root \
--network host \
--shm-size 4G \
-e ASCEND_VISIBLE_DEVICES=0,1,2,3 \
-v /etc/localtime:/etc/localtime \
-v /var/log/npu/:/usr/slog \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /data/containerd/workspace/:/workspace \
harbor.llm.io/base/llm-train-unify:v1-20240603 \
/bin/bash -c '. ~/.bashrc &&  conda activate llm-dev && sh /workspace/llm-train/scipts/local_run_unify_lora_npu.sh'
```

## 参考

- [Firefly](https://github.com/yangjianxin1/Firefly)







