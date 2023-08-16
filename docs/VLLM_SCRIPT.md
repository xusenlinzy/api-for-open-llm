## 环境配置

使用 `docker` 或者本地环境二者之一

### docker

```shell
docker build -f docker/Dockerfile.vllm -t llm-api:vllm .
```

### 本地环境

**`vLLM` 环境需要将 `torch` 版本升级到 `2.0.0` 以上，再安装 `vllm`**

```shell
pip install -r requirements.txt
pip install torch -U
pip install git+https://github.com/vllm-project/vllm.git
pip uninstall transformer-engine -y
```

## 启动模型

### 环境变量含义

+ `MODEL_NAME`: 模型名称，如 `qwen`、`baichuan-13b-chat` 等


+ `MODEL_PATH`: 开源大模型的文件所在路径


+ `TRUST_REMOTE_CODE`: 是否使用外部代码


+ `TOKENIZE_MODE`（可选项）: `tokenizer` 的模式，默认为 `auto`


+ `TENSOR_PARALLEL_SIZE`（可选项）: `GPU` 数量，默认为 `1`


+ `PROMPT_NAME`（可选项）: 使用的对话模板名称，如果不指定，则将根据模型名找到对应的模板


+ `EMBEDDING_NAME`（可选项）: 嵌入模型的文件所在路径，推荐使用 `moka-ai/m3e-base` 或者 `BAAI/bge-large-zh`


### 启动方式

选择下面两种方式之一启动模型接口服务

1. docker run

**不同模型只需要将 [.env.vllm.example](../.env.vllm.example) 文件内容复制到 `.env` 文件中，然后修改 `.env` 文件中环境变量**

```shell
docker run -it -d --gpus all --ipc=host -p 7891:8000 --name=vllm-server \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:vllm \
    python api/vllm_server.py
```

2. docker-compose

```shell
docker-compose -f docker-compose.vllm.yml up -d
```

**其中环境变量修改内容参考下面的模型**

### Qwen-7b-chat

Qwen/Qwen-7B-Chat:


```shell
MODEL_NAME=qwen
MODEL_PATH=Qwen/Qwen-7B-Chat # 模型所在路径，若使用docker，则为在容器内的路径
```

### InternLM

internlm-chat-7b:

```shell
MODEL_NAME=internlm
MODEL_PATH=internlm/internlm-chat-7b
```

### Baichuan-13b-chat

baichuan-inc/Baichuan-13B-Chat:

```shell
MODEL_NAME=baichuan-13b-chat
MODEL_PATH=baichuan-inc/Baichuan-13B-Chat
TENSOR_PARALLEL_SIZE=2
```