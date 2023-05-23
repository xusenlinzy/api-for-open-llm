# Api-for-Open-LLMs

开源大模型的统一后端接口，与 `OpenAI` 的响应保持一致

# 模型

支持多种开源大模型

+ [ChatGLM](https://github.com/THUDM/ChatGLM-6B)

+ [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

+ [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)

+ [MOSS](https://github.com/OpenLMLab/MOSS)

# 环境配置

## 本地启动

1. 搭建好 `pytorch` 深度学习环境

```shell
conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch cudatoolkit -c pytorch
```

2. 安装依赖包

```shell
pip install -r requirements.txt
```

## docker启动（**推荐**）

1. 构建镜像

```shell
docker build -t llm-api:pytorch .
```

2. 启动容器

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=chatglm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python main.py \
    --port 80 \
    --allow-credentials \
    --model_path THUDM/chatglm-6b \
    --embedding_name GanymedeNil/text2vec-large-chinese
```

主要参数含义：

+ `model_path`: 开源大模型的文件路径

+ `embedding_name`: 嵌入模型的文件路径

# 使用方式

## 环境变量

+ `OPENAI_API_KEY`: 此处随意填一个字符串即可

+ `OPENAI_API_BASE`: 后端启动的接口地址，如：http://192.168.0.25:80/v1


## 可接入的项目：

**几乎大部分的 `chatgpt` 应用和前后端项目都可以无缝衔接！**

+ [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web)

+ [dify](https://github.com/langgenius/dify)
