## RAG

### 环境变量配置示例

```shell
PORT=8000

# llm related
MODEL_NAME=qwen2
MODEL_PATH=Qwen/Qwen1.5-14B-Chat-GPTQ

# rag model related
EMBEDDING_NAME=maidalun1020/bce-embedding-base_v1
RERANK_NAME=maidalun1020/bce-reranker-base_v1

# vllm related
ENGINE=vllm
TOKENIZE_MODE=auto
gpu_memory_utilization=0.8
TENSOR_PARALLEL_SIZE=1
DTYPE=auto

TASKS=llm,rag
```

### 相关接口说明文档

待完善

### 快速体验

启动模型服务之后

```shell
cd streamlit-demo
```

修改 `.env` 文件中的以下两个变量

```shell
CHAT_API_BASE  # 聊天接口地址
EMBEDDING_API_BASE  # 嵌入模型接口地址（可选）
```

启动前端

```shell
streamlit run streamlit_app.py
```
