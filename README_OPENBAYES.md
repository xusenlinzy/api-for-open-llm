## 环境配置

### 本地环境

安装依赖，确保安装顺序严格按照下面的命令：

```shell
pip install vllm>=0.4.3
pip install -r requirements.txt 
```

## 启动模型

### 环境变量含义


+ `MODEL_NAME`: 模型名称，如 `chatglm4`、`qwen2`、`llama3`等


+ `PROMPT_NAME`: 使用的对话模板名称，如果不指定，则将根据 `tokenizer` 找到对应的模板


+ `MODEL_PATH`: 开源大模型的文件所在路径


+ `TRUST_REMOTE_CODE`: 是否使用外部代码


+ `TOKENIZE_MODE`（可选项）: `tokenizer` 的模式，默认为 `auto`


+ `TENSOR_PARALLEL_SIZE`（可选项）: `GPU` 数量，默认为 `1`


+ `EMBEDDING_NAME`（可选项）: 嵌入模型的文件所在路径，推荐使用 `moka-ai/m3e-base` 或者 `BAAI/bge-large-zh`


+ `GPU_MEMORY_UTILIZATION`（可选项）: `GPU` 占用率


+ `MAX_NUM_BATCHED_TOKENS`（可选项）: 每个批处理的最大 `token` 数量


+ `MAX_NUM_SEQS`（可选项）: 批量大小


+ `TASKS`（可选项）: `llm` 表示启动对话大模型，`rag` 表示启动文档文档相关接口，比如`embedding`、`rerank`


### 启动方式

#### 本地启动

根据需求修改 `.env` 文件中的环境变量

```shell
python server.py
```
#### 调用样例
```shell
curl -X POST "http://127.0.0.1:8080/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_API_KEY" \
-d '{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图像有什么东西？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/ByungKwanLee/TroL/blob/master/figures/demo.png?raw=true"
                    }
                }
            ]
        }
    ],
    "model": "minicpm-v"
}'
```
### 说明
目前只支持minicpm-v模型，下面是最大并发量测试结果：  
GPU_MEMORY_UTILIZATION=0.9 并发量10  
GPU_MEMORY_UTILIZATION=0.8 并发量14  
GPU_MEMORY_UTILIZATION=0.7 并发量20  
GPU_MEMORY_UTILIZATION=0.6 并发量28  
GPU_MEMORY_UTILIZATION=0.5 并发量30  
GPU_MEMORY_UTILIZATION=0.4 并发量36
