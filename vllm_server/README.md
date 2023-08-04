## 部署方式

### 安装依赖

```shell
pip install vllm
```

### 参数说明

- `model`: 开源大模型的文件所在路径


- `tokenizer-mode`: `tokenizer` 的模式


- `tensor_parallel_size` : 使用的 `GPU` 数量，默认为 `1`


- `model_name`: 模型名称


- `embedding_name`: 嵌入模型的文件所在路径


- `prompt_name`: 使用的对话模板名称，如果不指定，则将根据模型名找到对应的模板



### 启动脚本

baichuan-13b-chat

```shell
python vllm_server/api_server.py \
    --model checkpoints/baichuan-13b-chat \
    --model_name baichuan-13b-chat \
    --prompt_name baichuan \
    --tensor_parallel_size 2
```

llama2-7b-chat

```shell
python vllm_server/api_server.py \
    --model checkpoints/llama-2-7b-chat \
    --tokenizer-mode slow \
    --model_name llama-2-7b-chat \
    --prompt_name llama2 \
    --port 7891
```

