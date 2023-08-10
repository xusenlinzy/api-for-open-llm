## docker 镜像

构建镜像

```shell
docker build -f docker/Dockerfile.vllm -t llm-api:vllm .
```

## docker 启动模型

### 主要参数

+ `model_name`: 模型名称，如 `qwen`、`baichuan-13b-chat` 等


+ `model`: 开源大模型的文件所在路径


+ `trust-remote-code`: 是否使用外部代码


+ `tokenizer-mode`（可选项）: `tokenizer` 的模式


+ `embedding_name`（可选项）: 嵌入模型的文件所在路径


+ `prompt_name`（可选项）: 使用的对话模板名称，如果不指定，则将根据模型名找到对应的模板


### Qwen-7b-chat

Qwen/Qwen-7B-Chat:

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=qwen \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:vllm \
    python api/vllm_server.py \
    --port 80 \
    --allow-credentials \
    --model_name qwen \
    --model Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --tokenizer-mode slow
```
