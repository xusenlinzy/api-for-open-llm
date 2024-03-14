## 环境配置

使用 `docker` 或者本地环境两种方式之一，推荐使用 `docker`

### docker

构建镜像

```shell
docker build -f docker/Dockerfile.vllm -t llm-api:vllm .
```

### 本地环境

安装依赖，确保安装顺序严格按照下面的命令：

```shell
pip install torch==2.1.0
pip install vllm==0.3.3
pip install -r requirements.txt 
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


+ `GPU_MEMORY_UTILIZATION`（可选项）: `GPU` 占用率


+ `MAX_NUM_BATCHED_TOKENS`（可选项）: 每个批处理的最大 `token` 数量


+ `MAX_NUM_SEQS`（可选项）: 批量大小


### 启动方式

选择下面两种方式之一启动模型接口服务

#### docker启动

1. docker run

不同模型只需要将 [.env.vllm.example](../.env.vllm.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.vllm.example .env
```

然后修改 `.env` 文件中的环境变量

```shell
docker run -it -d --gpus all --ipc=host -p 7891:8000 --name=vllm-server \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:vllm \
    python api/server.py
```

2. docker compose

```shell
docker-compose -f docker-compose.vllm.yml up -d
```

#### 本地启动

同样的，将 [.env.vllm.example](../.env.vllm.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.vllm.example .env
```

然后修改 `.env` 文件中的环境变量

```shell
cp api/server.py .
python server.py
```

## 环境变量修改参考

**环境变量修改内容参考下面**

+ [internlm2](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#internlm2)    

+ [code-llama](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#code-llama) 

+ [sqlcoder](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#sqlcoder) 

+ [qwen-7b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#qwen-7b-chat)

+ [baichuan-13b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#baichuan-13b-chat)

+ [internlm](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md#internlm)      


### Qwen-7b-chat

Qwen/Qwen-7B-Chat:


```shell
MODEL_NAME=qwen
MODEL_PATH=Qwen/Qwen-7B-Chat # 模型所在路径，若使用docker，则为在容器内的路径
ENGINE=vllm
```

### InternLM

internlm-chat-7b:

```shell
MODEL_NAME=internlm
MODEL_PATH=internlm/internlm-chat-7b
ENGINE=vllm
```

### Baichuan-13b-chat

baichuan-inc/Baichuan-13B-Chat:

```shell
MODEL_NAME=baichuan-13b-chat
MODEL_PATH=baichuan-inc/Baichuan-13B-Chat
TENSOR_PARALLEL_SIZE=2
ENGINE=vllm
```

### SQLCODER

defog/sqlcoder:

```shell
MODEL_NAME=starcode
MODEL_PATH=defog/sqlcoder
TENSOR_PARALLEL_SIZE=2
ENGINE=vllm
```

### CODE-LLAMA

```shell
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/vllm-project/vllm.git
```

codellama/CodeLlama-7b-Instruct-hf

```shell
MODEL_NAME=code-llama
MODEL_PATH=codellama/CodeLlama-7b-Instruct-hf
ENGINE=vllm
```

### Xwin-LM

Xwin-LM/Xwin-LM-7B-V0.1

```shell
MODEL_NAME=xwin-7b
MODEL_PATH=Xwin-LM/Xwin-LM-7B-V0.1
PROMPT_NAME=vicuna
ENGINE=vllm
```

### InternLM2

internlm2-chat-20b:

```shell
MODEL_NAME=internlm2
MODEL_PATH=internlm/internlm2-chat-20b
ENGINE=vllm
TENSOR_PARALLEL_SIZE=2
```
