## 环境配置

使用 `docker` 或者本地环境二者之一

### docker

```shell
docker build -f docker/Dockerfile -t llm-api:pytorch .
```

### 本地环境

```shell
pip install torch==1.13.1
pip install -r requirements.txt
```

## 启动模型

### 环境变量含义

+ `MODEL_NAME`: 模型名称，如 `chatglm`、`phoenix`、`moss`等


+ `MODEL_PATH`: 开源大模型的文件所在路径


+ `DEVICE`: 是否使用 `GPU`，可选值为 `cuda` 和 `cpu`，默认值为 `cuda`


+ `ADAPTER_MODEL_PATH`（可选项）: `lora` 或 `ptuing_v2` 模型文件所在路径


+ `EMBEDDING_NAME`（可选项）: 嵌入模型的文件所在路径，推荐使用 `moka-ai/m3e-base` 或者 `BAAI/bge-large-zh`


+ `CONTEXT_LEN`（可选项）: 上下文长度，默认为 `2048`


+ `QUANTIZE`（可选项）: `chatglm`、`baichuan-13b` 模型的量化等级，可选项为 16、8、4


+ `LOAD_IN_8BIT`（可选项）: 使用模型 `8bit` 量化


+ `LOAD_IN_4BIT`（可选项）: 使用模型 `4bit` 量化


+ `USING_PTUNING_V2`（可选项）: 使用 `ptuning_v2` 加载模型


+ `STREAM_INTERVERL`（可选项）: 流式输出的 `token` 数量


+ `PROMPT_NAME`（可选项）: 使用的对话模板名称，如果不指定，则将根据模型名找到对应的模板


+ `PATCH_TYPE`（可选项）: 用来扩展 `llama` 模型上下文长度的长度，支持 `attention` 和 `ntk`


+ `ALPHA`（可选项）: 用来扩展 `llama` 模型上下文长度的长度，默认为 `auto`



### 启动方式

选择下面两种方式之一启动模型接口服务

1. docker run

**不同模型只需要将 [.env.example](../.env.example) 文件内容复制到 `.env` 文件中，然后修改 `.env` 文件中环境变量**

```shell
docker run -it -d --gpus all --ipc=host -p 7891:8000 --name=llm-api \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/server.py
```

2. docker-compose

```shell
docker-compose up -d
```

**其中环境变量修改内容参考下面的模型**

### ChatGLM

chatglm-6b:

```shell
MODEL_NAME=chatglm
MODEL_PATH=THUDM/chatglm-6b # 模型所在路径，若使用docker，则为在容器内的路径
```

chatglm2-6b:

```shell
MODEL_NAME=chatglm2
MODEL_PATH=THUDM/chatglm2-6b
```

ptuing-v2:

```shell
MODEL_NAME=chatglm
MODEL_PATH=THUDM/chatglm-6b
ADAPTER_MODEL_PATH=ptuing_v2_chekpint_dir
USING_PTUNING_V2=true
```

### MOSS

```shell
MODEL_NAME=moss
MODEL_PATH=fnlp/moss-moon-003-sft-int4
```

### Phoenix

```shell
MODEL_NAME=phoenix
MODEL_PATH=FreedomIntelligence/phoenix-inst-chat-7b
```

### Tiger

```shell
MODEL_NAME=tiger
MODEL_PATH=TigerResearch/tigerbot-7b-sft
```

### OpenBuddy

LLaMA

```shell
MODEL_NAME=openbuddy-llama
MODEL_PATH=OpenBuddy/openbuddy-llama-7b-v1.4-fp16
```

Falcon

```shell
MODEL_NAME=openbuddy-falcon
MODEL_PATH=OpenBuddy/openbuddy-falcon-7b-v5-fp16
```

### Baichuan-7b

使用半精度加载模型（大约需要14G显存）

```shell
MODEL_NAME=baichuan-7b
MODEL_PATH=baichuan-inc/baichuan-7B
ADAPTER_MODEL_PATH=YeungNLP/firefly-baichuan-7b-qlora-sft
```

### Baichuan-13b-chat

```shell
MODEL_NAME=baichuan-13b-chat
MODEL_PATH=baichuan-inc/Baichuan-13B-Chat
DEVICE_MAP=auto
```

可以使用 `QUANTIZE` 参数进行量化，例如 `QUANTIZE=8`


### InternLM

internlm-chat-7b:

```shell
MODEL_NAME=internlm
MODEL_PATH=internlm/internlm-chat-7b
```

### StarChat

starchat-beta:

```shell
MODEL_NAME=starchat
MODEL_PATH=HuggingFaceH4/starchat-beta
LOAD_IN_8BIT=true
```

### AquilaChat-7B

aquila-chat-7b:

```shell
MODEL_NAME=aquila-chat-7b
MODEL_PATH=BAAI/AquilaChat-7B
```

### Qwen-7b-chat

除已有的环境之外，推荐安装下面的依赖以提高运行效率和降低显存占用

```shell
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .

pip install csrc/layer_norm
pip install csrc/rotary
```

Qwen/Qwen-7B-Chat:

```shell
MODEL_NAME=qwen
MODEL_PATH=Qwen/Qwen-7B-Chat
DEVICE_MAP=auto
```


### XVERSE-13B-Chat

xverse/XVERSE-13B-Chat:

```shell
MODEL_NAME=xverse
MODEL_PATH=xverse/XVERSE-13B-Chat
DEVICE_MAP=auto
```

### SQLCODER

defog/sqlcoder:

```shell
MODEL_NAME=starcode
MODEL_PATH=defog/sqlcoder
DEVICE_MAP=auto
# LOAD_IN_8BIT=true
```

### CODE-LLAMA

TheBloke/CodeLlama-7B-Instruct-fp16

```shell
MODEL_NAME=code-llama
MODEL_PATH=TheBloke/CodeLlama-7B-Instruct-fp16
```