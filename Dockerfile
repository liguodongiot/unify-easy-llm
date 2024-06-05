
FROM docker.io/pepesi/ascend-base:ubuntu2204-py39-cann8-210

LABEL maintainer='guodong.li'

ENV APP_DIR=/workspaces
RUN mkdir -p -m 777 $APP_DIR

COPY llm-train ${APP_DIR}/llm-train

RUN cd ${APP_DIR}/llm-train && mv .bashrc ~/.bashrc && . ~/.bashrc \
&& conda create -n llm-dev python=3.9 && conda activate llm-dev \
&& pip install --no-cache-dir -r ${APP_DIR}/llm-train/requirements-npu-py39-torch210.txt \
&& rm -rf ~/.cache/pip/* && conda clean -all

