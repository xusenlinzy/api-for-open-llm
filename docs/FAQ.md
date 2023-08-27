# FAQ

## 安装&环境

###  docker 环境

构建镜像的命令为

```shell
docker build -f docker/Dockerfile -t llm-api:pytorch .
```

如果想要提高推理效率和处理并发请求，推荐使用 [vLLM](https://github.com/vllm-project/vllm) 

构建镜像的命令为

```shell
docker build -f docker/Dockerfile.vllm -t llm-api:vllm .
```

### 本地环境

**`vLLM` 环境需要将 `torch` 版本升级到 `2.0.0` 以上，再安装 `vllm`**

```shell
pip install -r requirements.txt
pip install torch -U
pip install vllm>=0.1.4
# pip install git+https://github.com/vllm-project/vllm.git
pip uninstall transformer-engine
```

如不需要安装 `vLLM`，则只需要

```shell
pip install torch==1.13
pip install -r requirements.txt
```

## 模型启动命令

### 不使用 vllm

模型启动命令及参数含义见 [script](./SCRIPT.md)

### 使用 vllm

模型启动命令及参数含义见 [vllm_script](./VLLM_SCRIPT.md)

**vllm 环境下 `embedding` 模型启动貌似会出问题**

解决方案：

```shell
pip uninstall transformer-engine
```

### 模型挂载

如果使用 `docker` 方式启动模型，且模型权重不在该项目下，需要将模型权重挂载到容器中，添加如下命令

```shell
-v {local_model_path}:/workspace/{container_model_path}
```

## 接口调用方式

### 可用的接口

+ `/v1/models`: 查询模型信息


+ `/v1/chat/completions`: 聊天


+ `/v1/completions`: 文字接龙


+ `/v1/embeddings`: 句子嵌入


具体使用方式兼容 [openai](https://github.com/openai/openai-python)

### 接入其他 `web` 项目

接入到其他基于 `chatgpt` 的前后端项目，只需要修改环境变量


+ `OPENAI_API_KEY`: 此处随意填一个字符串即可


+ `OPENAI_API_BASE`: 后端启动的接口地址，如：http://192.168.0.xx:80/v1
