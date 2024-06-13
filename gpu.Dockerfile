FROM nvcr.io/nvidia/pytorch:24.05-py3

LABEL maintainer='guodong.li'

ENV APP_DIR=/app
WORKDIR ${APP_DIR}
# RUN mkdir -p -m 777 $APP_DIR

COPY requirements-gpu-py310-torch210.txt ${APP_DIR}/
COPY dependences/pip.conf /usr/pip.conf

# ulimit -n 65535
RUN pip install --prefix /usr/local/py-env-low transformers==4.33.1 peft==0.6.0 \
&& pip install --no-cache-dir -r requirements-gpu-py310-torch210.txt \
&& rm -rf ~/.cache/pip/*

COPY . ${APP_DIR}/


