# API for Open LLMs

开源大模型的统一后端接口，与 `OpenAI` 的响应保持一致

# 🐼 模型

支持多种开源大模型

+ [ChatGLM](https://github.com/THUDM/ChatGLM-6B)

+ [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

+ [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)

+ [MOSS](https://github.com/OpenLMLab/MOSS)

# 🐳 环境配置

## 1. docker启动（**推荐**）

构建镜像

```shell
docker build -t llm-api:pytorch .
```

启动容器

```shell
docker run -it -d --gpus all --ipc=host --net=host -p 80:80 --name=chatglm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v `pwd`:/workspace \
    llm-api:pytorch \
    python main.py \
    --port 80 \
    --allow-credentials \
    --model_name chatglm \
    --model_path THUDM/chatglm-6b \
    --embedding_name GanymedeNil/text2vec-large-chinese
```

主要参数含义：

+ `model_name`: 模型名称，如`chatglm`、`phoenix`、`moss`等

+ `model_path`: 开源大模型的文件所在路径

+ `embedding_name`: 嵌入模型的文件所在路径

## 2. 本地启动

安装 `pytorch` 环境

```shell
conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch cudatoolkit -c pytorch
```

安装依赖包

```shell
pip install -r requirements.txt
```

启动后端

```shell
python main.py \
    --port 80 \
    --allow-credentials \
    --model_path THUDM/chatglm-6b \
    --embedding_name GanymedeNil/text2vec-large-chinese
```

# 🤖 使用方式

## 1. 环境变量

+ `OPENAI_API_KEY`: 此处随意填一个字符串即可

+ `OPENAI_API_BASE`: 后端启动的接口地址，如：http://192.168.0.xx:80/v1


## 2. [openai-python](https://github.com/openai/openai-python)

```python
import openai

# Point requests to Basaran by overwriting openai.api_base.
# Or you can use the OPENAI_API_BASE environment variable instead.
openai.api_base = "http://192.168.0.xx:80/v1"

# Enter any non-empty API key to pass the client library's check.
openai.api_key = "xxx"

# Enter any non-empty model name to pass the client library's check.
completion = openai.ChatCompletion.create(
    model="chatglm-6b",
    messages=[
        {"role": "user", "content": "你好"},
    ],
    stream=False,
)

print(completion.choices[0].message.content)
# 你好，有什么我可以帮助你的吗？
```

## 3. [langchain](https://github.com/hwchase17/langchain)

```python
import os

os.environ["OPENAI_API_BASE"] = "http://192.168.0.xx:80/v1"
os.environ["OPENAI_API_KEY"] = "xxx"

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)

chat = ChatOpenAI()
print(chat([HumanMessage(content="你好")]))
# content='你好，有什么我可以帮助您的吗？' additional_kwargs={}
```

## 4. 可接入的项目

**通过修改上面的 `OPENAI_API_BASE` 环境变量，大部分的 `chatgpt` 应用和前后端项目都可以无缝衔接！**

+ [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web)

```shell
docker run -d -p 3000:3000 \
   -e OPENAI_API_KEY="sk-xxxx" \
   -e BASE_URL="http://192.168.0.xx:80" \
   yidadaa/chatgpt-next-web
```

![web](images/web.png)

+ [dify](https://github.com/langgenius/dify)

![dify](images/dify.png)


