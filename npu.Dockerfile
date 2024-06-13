FROM docker.io/pepesi/ascend-base:ubuntu2204-py39-cann8-210

LABEL maintainer='guodong.li'

ENV APP_DIR=/app
#RUN mkdir -p -m 777 $APP_DIR

RUN rm -rf /bin/sh && ln -sf /bin/bash /bin/sh

COPY dependences/.bashrc /root/.bashrc
COPY requirements-npu-py39-torch210.txt $APP_DIR/

RUN . /root/.bashrc \
&& conda create -n llm-dev python=3.9 && conda activate llm-dev \
&& pip install --no-cache-dir -r ${APP_DIR}/llm-train/requirements-npu-py39-torch210.txt \
&& rm -rf ~/.cache/pip/* && conda clean -all

COPY . ${APP_DIR}/
