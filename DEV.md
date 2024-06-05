

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
docker build --network=host -f llm-train/train-env.Dockerfile -t harbor.llm.io/base/llm-train-unify:v1-20240603 .

```

## 启动训练任务

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