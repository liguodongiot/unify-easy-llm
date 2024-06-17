ARG BASE_IMG
FROM ${BASE_IMG}

ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    PYTHONUNBUFFERED=1

ARG PYVERSION

ENV PYVERSION=${PYVERSION}

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev gcc

RUN wget --no-check-certificate -O /tmp/Python-${PYVERSION}.tar.xz https://www.python.org/ftp/python/${PYVERSION}/Python-${PYVERSION}.tar.xz \
    && cd /tmp \
    && tar xvf Python-${PYVERSION}.tar.xz \
    && cd /tmp/Python-${PYVERSION} \
    && ./configure --enable-optimizations \
    && make \
    && make install \
    && cd /tmp \
    && rm -fr Python-${PYVERSION} Python-${PYVERSION}.tar.xz

RUN apt-get autoremove -y

ARG CANN_TOOLKIT_URL
ENV CANN_TOOLKIT_URL=${CANN_TOOLKIT_URL}
RUN wget -nv --no-check-certificate -O /tmp/toolkit.run ${CANN_TOOLKIT_URL} \
    && chmod +x /tmp/toolkit.run \
    && /tmp/toolkit.run --quiet --install --install-for-all \
    && rm -f /tmp/toolkit.run


ARG CANN_KERNEL_URL
ENV CANN_KERNEL_URL=${CANN_KERNEL_URL}
RUN wget -nv --no-check-certificate -O /tmp/kernel.run ${CANN_KERNEL_URL} \
    && chmod +x /tmp/kernel.run \
    && /tmp/kernel.run  --quiet --install --install-for-all \
    && rm -f /tmp/kernel.run

ARG TORCH_VERSION
ENV TORCH_VERSION=${TORCH_VERSION}

ARG TORCH_NPU_VERSION
ENV TORCH_NPU_VERSION=${TORCH_NPU_VERSION}

RUN apt-get install -y python3-pip && pip3 install -U pip -i https://mirrors.cloud.tencent.com/pypi/simple
RUN python3 -m pip install  pyyaml setuptools numpy \
 scipy psutil attrs decorator tornado  \
 torch==${TORCH_VERSION} torch-npu==${TORCH_NPU_VERSION} \
 -i https://mirrors.cloud.tencent.com/pypi/simple

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
ENV LD_PRELOAD=$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libgomp.so.1

RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" > ~/.bashrc
