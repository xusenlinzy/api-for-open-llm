## 环境配置

使用 `docker` 或者本地环境两种方式之一，推荐使用 `docker`

### docker

构建镜像

```shell
docker build -f docker/Dockerfile -t llm-api:pytorch .
```

### 本地环境

安装依赖

```shell
pip install torch>=2.3.0
pip install -r requirements.txt
```

## 启动模型

### 环境变量含义

+ `MODEL_NAME`: 模型名称，如 `chatglm4`、`qwen2`、`llama3`等


+ `PROMPT_NAME`: 使用的对话模板名称，如果不指定，则将根据 `tokenizer` 找到对应的模板


+ `MODEL_PATH`: 开源大模型的文件所在路径


+ `EMBEDDING_NAME`（可选项）: 嵌入模型的文件所在路径，推荐使用 `moka-ai/m3e-base` 或者 `BAAI/bge-large-zh`


+ `CONTEXT_LEN`（可选项）: 上下文长度，默认为 `2048`


+ `LOAD_IN_8BIT`（可选项）: 使用模型 `8bit` 量化


+ `LOAD_IN_4BIT`（可选项）: 使用模型 `4bit` 量化


+ `TASKS`（可选项）: `llm` 表示启动对话大模型，`rag` 表示启动文档文档相关接口，比如`embedding`、`rerank`


### 启动方式

选择下面两种方式之一启动模型接口服务


#### docker启动

1. docker run

不同模型只需要将 [.env.example](../.env.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.example .env
```

然后修改 `.env` 文件中的环境变量


```shell
docker run -it -d --gpus all --ipc=host -p 7891:8000 --name=llm-api \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/server.py
```

2. docker compose

```shell
docker-compose up -d
```

#### 本地启动

同样的，将 [.env.example](../.env.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.example .env
```

然后修改 `.env` 文件中的环境变量

```shell
cp api/server.py .
python server.py
```


## 环境变量修改参考

### QWEN系列

| 模型      | 环境变量示例                                                                                          |
|---------|-------------------------------------------------------------------------------------------------|
| qwen    | `MODEL_NAME=qwen`、`MODEL_PATH=Qwen/Qwen-7B-Chat`、`PROMPT_NAME=qwen`、 `DEVICE_MAP=cuda:0`        |
| qwen1.5 | `MODEL_NAME=qwen2`、`MODEL_PATH=Qwen/Qwen1.5-7B-Chat`、`PROMPT_NAME=qwen2`、 `DEVICE_MAP=cuda:0`   |
| qwen2   | `MODEL_NAME=qwen2`、`MODEL_PATH=Qwen/Qwen2-7B-Instruct`、`PROMPT_NAME=qwen2`、 `DEVICE_MAP=cuda:0` |


### GLM系列

| 模型        | 环境变量示例                                                                                                     |
|-----------|------------------------------------------------------------------------------------------------------------|
| chatglm   | `MODEL_NAME=chatglm`、`MODEL_PATH=THUDM/chatglm-6b`、`PROMPT_NAME=chatglm`、 `DEVICE_MAP=cuda:0`              |
| chatglm2  | `MODEL_NAME=chatglm2`、`MODEL_PATH=THUDM/chatglm2-6b`、`PROMPT_NAME=chatglm2`、 `DEVICE_MAP=cuda:0`           |
| chatglm3  | `MODEL_NAME=chatglm3`、`MODEL_PATH=THUDM/chatglm3-6b`、`PROMPT_NAME=chatglm3`、 `DEVICE_MAP=cuda:0`           |
| glm4-chat | `MODEL_NAME=chatglm4`、`MODEL_PATH=THUDM/glm-4-9b-chat`、`PROMPT_NAME=chatglm4`、 `DEVICE_MAP=cuda:0`         |
| glm-4v    | `MODEL_NAME=glm-4v`、`MODEL_PATH=THUDM/glm-4v-9b`、`PROMPT_NAME=glm-4v`、 `DEVICE_MAP=auto`、 `DTYPE=bfloat16` |


### BAICHUAN系列

| 模型        | 环境变量示例                                                                                                         |
|-----------|----------------------------------------------------------------------------------------------------------------|
| baichuan  | `MODEL_NAME=baichuan`、`MODEL_PATH=baichuan-inc/Baichuan-13B-Chat`、`PROMPT_NAME=baichuan`、 `DEVICE_MAP=auto`    |
| baichuan2 | `MODEL_NAME=baichuan2`、`MODEL_PATH=baichuan-inc/Baichuan2-13B-Chat`、`PROMPT_NAME=baichuan2`、 `DEVICE_MAP=auto` |


### INTERNLM系列

| 模型        | 环境变量示例                                                                                                     |
|-----------|------------------------------------------------------------------------------------------------------------|
| internlm  | `MODEL_NAME=internlm`、`MODEL_PATH=internlm/internlm-chat-7b`、`PROMPT_NAME=internlm`、 `DEVICE_MAP=cuda:0`   |
| internlm2 | `MODEL_NAME=internlm2`、`MODEL_PATH=internlm/internlm2-chat-20b`、`PROMPT_NAME=internlm2`、 `DEVICE_MAP=auto` |


### Yi系列

| 模型    | 环境变量示例                                                                                    |
|-------|-------------------------------------------------------------------------------------------|
| yi    | `MODEL_NAME=yi`、`MODEL_PATH=01-ai/Yi-34B-Chat`、`PROMPT_NAME=yi`、 `DEVICE_MAP=auto`        |
| yi1.5 | `MODEL_NAME=yi1.5`、`MODEL_PATH=01-ai/Yi1.5-9B-Chat`、`PROMPT_NAME=yi`、 `DEVICE_MAP=cuda:0` |


### DEEPSEEK系列

| 模型             | 环境变量示例                                                                                                                           |
|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| deepseek-coder | `MODEL_NAME=deepseek-coder`、`MODEL_PATH=deepseek-ai/deepseek-coder-33b-instruct`、`PROMPT_NAME=deepseek-coder`、 `DEVICE_MAP=auto` |
| deepseek-chat  | `MODEL_NAME=deepseek`、`MODEL_PATH=deepseek-ai/deepseek-llm-67b-chat`、`PROMPT_NAME=deepseek`、 `DEVICE_MAP=auto`                   |


### LLAMA系列

| 模型     | 环境变量示例                                                                                                         |
|--------|----------------------------------------------------------------------------------------------------------------|
| llama2 | `MODEL_NAME=llama2`、`MODEL_PATH=meta-llama/Llama-2-7b-chat-hf`、`PROMPT_NAME=llama2`、 `DEVICE_MAP=cuda:0`       |
| llama3 | `MODEL_NAME=llama3`、`MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct`、`PROMPT_NAME=llama3`、 `DEVICE_MAP=cuda:0` |
