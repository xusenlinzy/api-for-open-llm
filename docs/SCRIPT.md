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
pip install torch>=1.13.1
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


+ `PATCH_TYPE`（可选项）: 用来扩展 `llama` 模型上下文长度，支持 `attention` 和 `ntk`


+ `ALPHA`（可选项）: 用来扩展 `llama` 模型上下文长度，默认为 `auto`



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

**环境变量修改内容参考下面**

+ [internlm2](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#internlm2)  

+ [sus-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#suschat)

+ [deepseek](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#deepseekchat)

+ [deepseek-coder](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#deepseekcoder)

+ [yi-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#yi-chat)

+ [baichuan2](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#baichuan2)

+ [code-llama](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#code-llama)

+ [sqlcoder](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#sqlcoder)  

+ [xverse-13b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#xverse-13b-chat) 

+ [qwen-7b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#qwen-7b-chat)

+ [aquila-chat-7b](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#aquilachat-7b)  

+ [starchat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#starchat)       

+ [baichuan-13b-chat](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#baichuan-13b-chat) 

+ [internlm](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#internlm)      

+ [baichuan-7b](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#baichuan-7b)    

+ [openbuddy](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#openbuddy)      

+ [chatglm](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#chatglm)        

+ [moss](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#moss)         

+ [phoenix](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#phoenix)     

+ [tiger](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#tiger)       

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

### ChatGLM3:

```shell
MODEL_NAME=chatglm3
MODEL_PATH=THUDM/chatglm3-6b
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

### Qwen-14b-chat

Qwen/Qwen-14B-Chat:

```shell
MODEL_NAME=qwen
MODEL_PATH=Qwen/Qwen-14B-Chat
DEVICE_MAP=auto
```

Qwen/Qwen-14B-Chat-Int4:

本地环境安装下面的依赖包

```shell
pip install auto-gptq optimum
```

`docker` 环境使用下面的命令构建一个新的 `GPTQ` 镜像，并基于此镜像启动模型

```shell
docker build -f docker/Dockerfile.gptq -t llm-api:gptq .
```

```shell
MODEL_NAME=qwen
MODEL_PATH=Qwen/Qwen-14B-Chat-Int4
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

codellama/CodeLlama-7b-Instruct-hf

```shell
MODEL_NAME=code-llama
MODEL_PATH=codellama/CodeLlama-7b-Instruct-hf
```

### Wizard-Coder

WizardLM/WizardCoder-Python-34B-V1.0

```shell
MODEL_NAME=code-llama
MODEL_PATH=WizardLM/WizardCoder-Python-34B-V1.0
PROMPT_NAME=alpaca
DEVICE_MAP=auto
```


### Baichuan2

`Baichuan2` 系列模型中，为了加快推理速度使用了 `pytorch2.0` 加入的新功能 `F.scaled_dot_product_attention`，因此需要在 `pytorch2.0` 环境下运行

可以使用下面的命令升级 `llm-api:pytorch` 环境，或者直接使用 `llm-api:vllm` 环境

```shell
pip install torch -U
# pip uninstall transformer-engine -y
```

baichuan-inc/Baichuan2-13B-Chat

```shell
MODEL_NAME=baichuan2-13b-chat
MODEL_PATH=baichuan-inc/Baichuan2-13B-Chat
DEVICE_MAP=auto
DTYPE=bfloat16
```

`BitsAndBytes` 量化

```shell
MODEL_NAME=baichuan2-13b-chat
MODEL_PATH=baichuan-inc/Baichuan2-13B-Chat
DEVICE_MAP=auto
LOAD_IN_8BIT=true
```

在线量化

```shell
MODEL_NAME=baichuan2-13b-chat
MODEL_PATH=baichuan-inc/Baichuan2-13B-Chat
DEVICE_MAP=
DTYPE=half
QUANTIZE=8
```

### Xwin-LM

Xwin-LM/Xwin-LM-7B-V0.1

```shell
MODEL_NAME=xwin-7b
MODEL_PATH=Xwin-LM/Xwin-LM-7B-V0.1
PROMPT_NAME=vicuna
```

### XuanYuan

Duxiaoman-DI/XuanYuan-70B-Chat

```shell
MODEL_NAME=llama2
MODEL_PATH=Duxiaoman-DI/XuanYuan-70B-Chat
PROMPT_NAME=xuanyuan
```

### Yi-Chat

01-ai/Yi-34B-Chat

```shell
MODEL_NAME=yi-chat
MODEL_PATH=01-ai/Yi-34B-Chat
PROMPT_NAME=yi
DEVICE_MAP=auto
```

### DeepSeekCoder

deepseek-ai/deepseek-coder-33b-instruct

```shell
MODEL_NAME=deepseek-coder
MODEL_PATH=deepseek-ai/deepseek-coder-33b-instruct
DEVICE_MAP=auto
```


### DeepseekChat

deepseek-ai/deepseek-llm-67b-chat

```shell
MODEL_NAME=deepseek
MODEL_PATH=deepseek-ai/deepseek-llm-67b-chat
DEVICE_MAP=auto
```

### SUSChat

SUSTech/SUS-Chat-34B

```shell
MODEL_NAME=sus-chat
MODEL_PATH=SUSTech/SUS-Chat-34B
DEVICE_MAP=auto
```

### InternLM2

internlm2-chat-20b:

```shell
MODEL_NAME=internlm2
MODEL_PATH=internlm/internlm2-chat-20b
DEVICE_MAP=auto
```


