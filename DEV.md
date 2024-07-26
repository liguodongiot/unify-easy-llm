



## 本地环境


```
export BASE_CODE_PATH="/home/guodong.li"
export LOCAL_TEMP_DIR=$BASE_CODE_PATH/workspace/temp
export LOCAL_DATASET_PATH="$LOCAL_TEMP_DIR/datas"
export LOCAL_MODEL_PATH="$LOCAL_TEMP_DIR/models"
export LOCAL_OUTPUT_PATH="$LOCAL_TEMP_DIR/outputs"
export LOCAL_LOG_PATH="$LOCAL_TEMP_DIR/logs"
# lora
export LOCAL_MERGE_PATH="$LOCAL_TEMP_DIR/merges"
# 进度
export LOCAL_PROGRESS_PATH=$LOCAL_OUTPUT_PATH"/progress.json"


#rm -rf $LOCAL_TEMP_DIR
mkdir -p $LOCAL_DATASET_PATH
mkdir -p $LOCAL_MODEL_PATH
mkdir -p $LOCAL_OUTPUT_PATH
mkdir -p $LOCAL_LOG_PATH
# lora
mkdir -p $LOCAL_MERGE_PATH

```





## Docker

```
docker pull docker.io/pepesi/ascend-base:ubuntu2204-py39-cann8-210

docker rm -f  pytorch_dev

docker run -it -u root \
--name pytorch_dev \
--network host \
--shm-size 4G \
-e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
-v /etc/localtime:/etc/localtime \
-v /var/log/npu/:/usr/slog \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /data/containerd/workspace/:/workspace \
docker.io/pepesi/ascend-base:ubuntu2204-py39-cann8-210 \
/bin/bash

docker start pytorch_dev
docker exec -it pytorch_dev bash
```


```
docker pull nvcr.io/nvidia/pytorch:24.05-py3

docker rm -f  pytorch_cuda_dev

sudo docker run -it --gpus '"device=5"' \
--name pytorch_cuda_dev \
--shm-size 4G \
-w /workspace \
--env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
nvcr.io/nvidia/pytorch:24.05-py3 \
/bin/bash

docker start pytorch_cuda_dev
docker exec -it pytorch_cuda_dev bash
```


## Conda 
```
conda init

. /root/.bashrc 

conda create -n llm-dev python=3.9
conda activate llm-dev 
```

## 开发环境

```
#pip3 install torch==2.1.0 -i https://pypi.org/simple 
pip3 install torch==2.1.0  -i https://mirrors.cloud.tencent.com/pypi/simple
pip3 install pyyaml setuptools -i https://mirrors.cloud.tencent.com/pypi/simple
pip3 install torch-npu==2.1.0.post3 -i https://mirrors.cloud.tencent.com/pypi/simple
pip3 install numpy attrs decorator psutil absl-py cloudpickle psutil scipy synr tornado -i https://mirrors.cloud.tencent.com/pypi/simple


pip3 install multiprocessing inspect numpy attrs decorator psutil absl-py cloudpickle psutil scipy synr tornado -i https://mirrors.aliyun.com/pypi/simple
pip3 install torch==2.1.0  -i https://mirrors.aliyun.com/pypi/simple
pip3 install pyyaml setuptools -i https://mirrors.aliyun.com/pypi/simple
pip3 install torch-npu==2.1.0.post3 -i https://mirrors.aliyun.com/pypi/simple


pip install --no-cache-dir -r requirements-npu.txt -i https://mirrors.aliyun.com/pypi/simple
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```



## 构建镜像
```
cd /data/containerd/workspace/
docker build --network=host -f npu.Dockerfile -t harbor.llm.io/base/llm-train-unify:v1-20240603-cann8 .

```

```
docker build --network=host -f gpu.Dockerfile -t harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 .



sudo docker run -it --rm --gpus '"device=5"' \
--shm-size 4G \
-w /workspace \
--env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 \
/bin/bash
```


## 启动训练任务

```
deepspeed --num_gpus=1 train_lora.py --train_args_file $TRAIN_ARGS_PATH
```


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


```
docker build --network=host -f gpu.Dockerfile -t harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 .

sudo docker run -it --rm --gpus all \
--shm-size 4G \
-v /data/hpc/home/guodong.li:/home/guodong.li \
--env PYTHONPATH=/usr/local/py-env-low/local/lib/python3.10/dist-packages:$PYTHONPATH \
--env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
--env CUDA_VISIBLE_DEVICES=7 \
harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 \
/bin/bash /app/scripts/local_run_unify_sft_gpu.sh
```



```
docker build --network=host -f gpu.Dockerfile -t harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 .

sudo docker run -it --rm --gpus all \
--shm-size 4G \
-v /data/hpc/home/guodong.li:/home/guodong.li \
--env PYTHONPATH=/usr/local/py-env-low/local/lib/python3.10/dist-packages:$PYTHONPATH \
--env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
--env CUDA_VISIBLE_DEVICES=7 \
harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 \
/bin/bash /app/scripts/local_run_unify_lora_gpu.sh


```

glm4:

```
sudo docker run -it --rm --gpus '"device=6,7"' \
--shm-size 4G \
-v /data/hpc/home/guodong.li/workspace/temp:/home/guodong.li/workspace/temp \
harbor.llm.io/base/llm-train-unify:v1-20240603-cuda124 \
/bin/bash /app/scripts/local_run_unify_sft_gpu.sh
```


## CPU


```
conda activate pytorch-venv

sh scripts/local_run_unify_sft_cpu.sh

python train_unify.py --train_args_file /Users/liguodong/work/github/lgd/unify-easy-llm/train_args/sft-config-cpu.json



"deepspeed": "/Users/liguodong/work/github/lgd/unify-easy-llm/train_args/ds_z2_offload.json",

```


```
pip install py-cpuinfo 
sudo pip install deepspeed
```



## 推理

```
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```