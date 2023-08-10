## docker 镜像

构建镜像

```shell
docker build -f docker/Dockerfile -t llm-api:pytorch .
```

或者拉取已经构建好的镜像

```shell
docker pull xusenlinzy/llm-api:latest
```

**使用拉取的镜像需要将下面命令中的镜像名称替换成 `xusenlinzy/llm-api:latest`**

## docker 启动模型

### 主要参数

+ `model_name`: 模型名称，如 `chatglm`、`phoenix`、`moss`等


+ `model_path`: 开源大模型的文件所在路径


+ `device`: 是否使用 `GPU`，可选值为 `cuda` 和 `cpu`，默认值为 `cuda`


+ `adapter_model_path`（可选项）: `lora` 或 `ptuing_v2` 模型文件所在路径


+ `embedding_name`（可选项）: 嵌入模型的文件所在路径


+ `context_len`（可选项）: 上下文长度，默认为 `2048`


+ `quantize`（可选项）: `chatglm`、`baichuan-13b` 模型的量化等级，可选项为 16、8、4


+ `load_in_8bit`（可选项）: 使用模型 `8bit` 量化


+ `load_in_4bit`（可选项）: 使用模型 `4bit` 量化


+ `use_ptuning_v2`（可选项）: 使用 `ptuning_v2` 加载模型


+ `stream_interval`（可选项）: 流式输出的 `token` 数量


+ `prompt_name`（可选项）: 使用的对话模板名称，如果不指定，则将根据模型名找到对应的模板


+ `patch_type`（可选项）: 用来扩展 `llama` 模型上下文长度的长度，支持 [`ntk`](https://kexue.fm/archives/9706) 和 [`rerope`](https://spaces.ac.cn/archives/9708)


+ `training_length`（可选项）: 用来扩展 `llama` 模型上下文长度的长度，训练长度


+ `window_size`（可选项）: 用来扩展 `llama` 模型上下文长度的长度，窗口大小，小于训练长度


### ChatGLM

chatglm-6b:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=chatglm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name chatglm \
    --model_path THUDM/chatglm-6b \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

chatglm2-6b:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=chatglm2 \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name chatglm2 \
    --model_path THUDM/chatglm2-6b \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

ptuing-v2:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=chatglm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name chatglm \
    --model_path THUDM/chatglm-6b \
    --device cuda \
    --adapter_model_path ptuing_v2_chekpint_dir \
    --use_ptuning_v2 \
    --embedding_name moka-ai/m3e-base
```

### MOSS

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=moss \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name moss \
    --model_path fnlp/moss-moon-003-sft-int4 \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

### Phoenix

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=phoenix \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name phoenix \
    --model_path FreedomIntelligence/phoenix-inst-chat-7b \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

### Guanaco

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=guanaco \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name guanaco \
    --model_path timdettmers/guanaco-33b-merged \
    --device cuda \
    --load_in_4bit \
    --embedding_name moka-ai/m3e-base
```

### Tiger

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=tiger \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name tiger \
    --model_path TigerResearch/tigerbot-7b-sft \
    --device cuda \
    --load_in_4bit \
    --embedding_name moka-ai/m3e-base
```

### OpenBuddy

LLaMA

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=openbuddy-llama \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name openbuddy-llama \
    --model_path OpenBuddy/openbuddy-llama-7b-v1.4-fp16 \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

Falcon

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=openbuddy-falcon \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name openbuddy-falcon \
    --model_path OpenBuddy/openbuddy-falcon-7b-v5-fp16 \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

### Baichuan-7b

使用半精度加载模型（大约需要14G显存）

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=baichuan-7b \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name baichuan-7b \
    --model_path baichuan-inc/baichuan-7B \
    --adapter_model_path YeungNLP/firefly-baichuan-7b-qlora-sft \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

如果想要使用全精度加载模型，需要修改

```python
@property
def model_kwargs(self):
    return {"torch_dtype": torch.float32}
```


### Baichuan-13b-chat

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=baichuan-13b-chat \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name baichuan-13b-chat \
    --model_path baichuan-inc/Baichuan-13B-Chat \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

若模型加载到 `GPU` 太慢，可以通过修改

```python
@property
def model_kwargs(self):
    return {"device_map": "auto"}
```

可以使用 `quantize` 参数进行量化，例如 `--quantize 8`


### InternLM

internlm-chat-7b:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=internlm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name internlm \
    --model_path internlm/internlm-chat-7b \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

### StarChat

starchat-beta:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=starchat \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name starchat \
    --model_path HuggingFaceH4/starchat-beta \
    --device cuda \
    --load_in_8bit \
    --embedding_name moka-ai/m3e-base
```

### AquilaChat-7B

aquila-chat-7b:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=aquila-chat-7b \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name aquila-chat-7b \
    --model_path BAAI/AquilaChat-7B \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```

### NewHope

SLAM-group/NewHope:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=newhope \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name newhope \
    --model_path SLAM-group/NewHope \
    --device cuda \
    --embedding_name moka-ai/m3e-base
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
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=qwen \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python api/app.py \
    --port 80 \
    --allow-credentials \
    --model_name qwen \
    --model_path Qwen/Qwen-7B-Chat \
    --device cuda \
    --embedding_name moka-ai/m3e-base
```
