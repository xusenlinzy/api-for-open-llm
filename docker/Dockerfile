FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
COPY requirements.txt /workspace/
COPY . /workspace

RUN pip install --no-cache-dir -r /workspace/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install bitsandbytes --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple