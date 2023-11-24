## 环境配置

推荐使用 `docker`

### docker

构建镜像

```shell
docker build -f docker/Dockerfile.llama.cpp -t llm-api:llama-cpp .
```

## 启动模型

### 环境变量含义


+ `MODEL_NAME`: 模型名称


+ `MODEL_PATH`: 开源大模型的文件所在路径


+ `N_GPU_LAYERS`: -1 表示全部放到 `GPU`


+ `MAIN_GPU`: 使用哪块 `GPU`


### 启动方式


不同模型只需要将 [.env.example](../.env.example) 文件内容复制到 `.env` 文件中

```shell
cp .env.vllm.example .env
```

然后修改 `.env` 文件中的环境变量


```shell
docker-compose -f docker-compose-llama-cpp.yml up -d
```


## 环境变量修改参考

**环境变量修改内容参考下面**


### Baichuan2-7b-chat-gguf

baichuan2-7b-chat-gguf

```shell
MODEL_NAME=baichuan2
MODEL_PATH=checkpoints/baichuan2-7b-chat-gguf/baichuan2-7b-chat.Q3_K.gguf
ENGINE=llama.cpp
N_GPU_LAYERS=-1
```
