
docker build \
    --build-arg CANN_KERNEL_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC1/Ascend-cann-kernels-910b_8.0.RC1_linux.run" \
    --build-arg CANN_TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC1/Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run" \
    --build-arg PYVERSION=3.9.19 \
    --build-arg BASE_IMG=ubuntu:20.04 \
    --build-arg TORCH_VERSION=2.2.0 \
    --build-arg TORCH_NPU_VERSION=2.2.0 \
    --pull \
    -f Dockerfile.template \
    -t ${REPO}ascend-base:ubuntu2004-py39-cann8 .



    