# API for Open LLMs
  
## 🐧 QQ交流群：870207830

## 📢 新闻

+ 【2023.08.28】 添加 `transformers.TextIteratorStreamer` 流式输出支持，只需将环境变量修改为 `USE_STREAMER_V2=true`


+ 【2023.08.26】 添加 [code-llama](https://github.com/facebookresearch/codellama) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#code-llama)，[使用示例链接](https://github.com/xusenlinzy/api-for-open-llm/tree/master/examples/code-llama)


+ 【2023.08.23】 添加 [sqlcoder](https://huggingface.co/defog/sqlcoder) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#sqlcoder)，[使用示例链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/examples/sqlcoder/inference.py)


+ 【2023.08.22】 添加 [xverse-13b-chat](https://github.com/xverse-ai/XVERSE-13B) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#xverse-13b-chat)


+ 【2023.08.10】 添加 [vLLM](https://github.com/vllm-project/vllm) 推理加速支持，[使用文档](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md)


+ 【2023.08.03】 添加 [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B) 模型支持，[启动方式链接](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md#qwen-7b-chat)


更多新闻和历史请转至 [此处](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/NEWS.md)

---

**此项目主要内容**

此项目为开源大模型的推理实现统一的后端接口，与 `OpenAI` 的响应保持一致，具有以下特性：

+ ✨ 以 `OpenAI ChatGPT API` 的方式调用各类开源大模型


+ 🖨️ 支持流式响应，实现打印机效果


+ 📖 实现文本嵌入模型，为文档知识问答提供支持


+ 🦜️ 支持大规模语言模型开发工具 [`langchain` ](https://github.com/hwchase17/langchain) 的各类功能
 

+ 🙌 只需要简单的修改环境变量即可将开源模型作为 `chatgpt` 的替代模型，为各类应用提供后端支持


+ 🚀 支持加载经过自行训练过的 `lora` 模型


+ ⚡ 支持 [vLLM](https://github.com/vllm-project/vllm) 推理加速和处理并发请求


## 内容导引

|                                             章节                                              |            描述            |
|:-------------------------------------------------------------------------------------------:|:------------------------:|
|             [💁🏻‍♂支持模型](https://github.com/xusenlinzy/api-for-open-llm#-支持模型)              |     此项目支持的开源模型以及简要信息     |
|     [🚄启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md)     |      启动模型的环境配置和启动命令      |
| [⚡vLLM启动方式](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/VLLM_SCRIPT.md) | 使用 `vLLM` 启动模型的环境配置和启动命令 |
|               [💻调用方式](https://github.com/xusenlinzy/api-for-open-llm#-使用方式)                |       启动模型之后的调用方式        |
|       [❓常见问题](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/FAQ.md)       |        一些常见问题的回复         |
|   [📚相关资源](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/RESOURCES.md)    |     关于开源模型训练和推理的相关资源     |


## 🐼 支持模型

**语言模型**

|                                   模型                                   |     基座模型     |   参数量    |   语言   |                                                   模型权重链接                                                    |
|:----------------------------------------------------------------------:|:------------:|:--------:|:------:|:-----------------------------------------------------------------------------------------------------------:|
|       [codellama](https://github.com/facebookresearch/codellama)       |    LLaMA2    | 7/13/34B | multi  |       [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)       |
|       [xverse-13b-chat](https://github.com/xverse-ai/XVERSE-13B)       |    Xverse    |   13B    | multi  |                   [xverse/XVERSE-13B-Chat](https://huggingface.co/xverse/XVERSE-13B-Chat)                   |
|           [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B)            |     Qwen     |    7B    | en, zh |                 [Qwen/Qwen-7B-Chat](https://huggingface.co/baichuan-inc/Qwen/Qwen-7B-Chat)                  |
|   [baichuan-13b-chat](https://github.com/baichuan-inc/Baichuan-13B)    |   Baichuan   |   13B    | en, zh |           [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)           |
|            [InternLM](https://github.com/InternLM/InternLM)            |   InternLM   |    7B    | en, zh |                [internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)                |
|           [InternLM2](https://github.com/InternLM/InternLM)           |  InternLM2   |   20B    | en, zh |        [internlm/internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)                     |
|            [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)            |     GLM      |  6/130B  | en, zh |                        [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)                        |
|       [baichaun-7b](https://github.com/baichuan-inc/baichuan-7B)       |   Baichuan   |    7B    | en, zh |                 [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B)                 |
|         [Guanaco](https://github.com/artidoro/qlora/tree/main)         |    LLaMA     | 7/33/65B |   en   |           [timdettmers/guanaco-33b-merged](https://huggingface.co/timdettmers/guanaco-33b-merged)           |
|          [YuLan-Chat](https://github.com/RUC-GSAI/YuLan-Chat)          |    LLaMA     |  13/65B  | en, zh |            [RUCAIBox/YuLan-Chat-13b-delta](https://huggingface.co/RUCAIBox/YuLan-Chat-13b-delta)            |
|         [TigerBot](https://github.com/TigerResearch/TigerBot)          |    BLOOMZ    |  7/180B  | en, zh |            [TigerResearch/tigerbot-7b-sft](https://huggingface.co/TigerResearch/tigerbot-7b-sft)            |
|          [OpenBuddy](https://github.com/OpenBuddy/OpenBuddy)           | LLaMA、Falcon |    7B    | multi  |                                [OpenBuddy](https://huggingface.co/OpenBuddy)                                |
|               [MOSS](https://github.com/OpenLMLab/MOSS)                |   CodeGen    |   16B    | en, zh |              [fnlp/moss-moon-003-sft-int4](https://huggingface.co/fnlp/moss-moon-003-sft-int4)              |
|        [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)        |    BLOOMZ    |    7B    | multi  | [FreedomIntelligence/phoenix-inst-chat-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b) |
|        [BAIZE](https://github.com/project-baize/baize-chatbot)         |    LLaMA     | 7/13/30B |   en   |              [project-baize/baize-lora-7B](https://huggingface.co/project-baize/baize-lora-7B)              |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)  |    LLaMA     |  7/13B   | en, zh |   [ziqingyang/chinese-alpaca-plus-lora-7b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b)   |
|             [BELLE](https://github.com/LianjiaTech/BELLE)              |    BLOOMZ    |    7B    |   zh   |                   [BelleGroup/BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)                   |
|             [ChatGLM](https://github.com/THUDM/ChatGLM-6B)             |     GLM      |    6B    | en, zh |                         [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)                         |


**嵌入模型**

|           模型           |  维度  |                                        权重链接                                         |
|:----------------------:|:----:|:-----------------------------------------------------------------------------------:|
|      bge-large-zh      | 1024 |              [bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)               |
|       m3e-large        | 1024 |            [moka-ai/m3e-large](https://huggingface.co/moka-ai/m3e-large)            |
| text2vec-large-chinese | 1024 | [text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese) |


## 🤖 使用方式

### 环境变量

+ `OPENAI_API_KEY`: 此处随意填一个字符串即可

+ `OPENAI_API_BASE`: 后端启动的接口地址，如：http://192.168.0.xx:80/v1


### [聊天界面](https://github.com/xusenlinzy/api-for-open-llm/tree/master/streamlit-demo)

```shell
cd streamlit-demo
pip install -r requirements.txt
streamlit run streamlit_app.py
```


### [openai](https://github.com/openai/openai-python)

👉 Chat Completions

```python
import openai

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
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
```


👉 Completions

```python
import openai

openai.api_base = "http://192.168.0.xx:80/v1"

# Enter any non-empty API key to pass the client library's check.
openai.api_key = "xxx"

# Enter any non-empty model name to pass the client library's check.
completion = openai.Completion.create(prompt="你好", model="chatglm-6b")

print(completion.choices[0].text)
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
```


👉 Embeddings

```python
import openai

openai.api_base = "http://192.168.0.xx:80/v1"

# Enter any non-empty API key to pass the client library's check.
openai.api_key = "xxx"

# compute the embedding of the text
embedding = openai.Embedding.create(
    input="什么是chatgpt？", 
    model="text2vec-large-chinese"
)

print(embedding['data'][0]['embedding'])
```


### [langchain](https://github.com/hwchase17/langchain)

👉 Chat Completions

```python
import os

os.environ["OPENAI_API_BASE"] = "http://192.168.0.xx:80/v1"
os.environ["OPENAI_API_KEY"] = "xxx"

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI()
print(chat([HumanMessage(content="你好")]))
# content='你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。' additional_kwargs={}
```

>👉 Completions

```python
import os

os.environ["OPENAI_API_BASE"] = "http://192.168.0.xx:80/v1"
os.environ["OPENAI_API_KEY"] = "xxx"

from langchain.llms import OpenAI

llm = OpenAI()
print(llm("你好"))
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
```


👉 Embeddings


```python
import os

os.environ["OPENAI_API_BASE"] = "http://192.168.0.xx:80/v1"
os.environ["OPENAI_API_KEY"] = "xxx"

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
query_result = embeddings.embed_query("什么是chatgpt？")
print(query_result)
```


### 可接入的项目

**通过修改上面的 `OPENAI_API_BASE` 环境变量，大部分的 `chatgpt` 应用和前后端项目都可以无缝衔接！**

+ [ChatGPT-Next-Web: One-Click to deploy well-designed ChatGPT web UI on Vercel](https://github.com/Yidadaa/ChatGPT-Next-Web)

```shell
docker run -d -p 3000:3000 \
   -e OPENAI_API_KEY="sk-xxxx" \
   -e BASE_URL="http://192.168.0.xx:80" \
   yidadaa/chatgpt-next-web
```


+ [dify: An easy-to-use LLMOps platform designed to empower more people to create sustainable, AI-native applications](https://github.com/langgenius/dify)

```shell
# 在docker-compose.yml中的api和worker服务中添加以下环境变量
OPENAI_API_BASE: http://192.168.0.xx:80/v1
DISABLE_PROVIDER_CONFIG_VALIDATION: 'true'
```


## 📜 License

此项目为 `Apache 2.0` 许可证授权，有关详细信息，请参阅 [LICENSE](LICENSE) 文件。


## 🚧 References

+ [ChatGLM: An Open Bilingual Dialogue Language Model](https://github.com/THUDM/ChatGLM-6B)

+ [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

+ [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)

+ [Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

+ [Phoenix: Democratizing ChatGPT across Languages](https://github.com/FreedomIntelligence/LLMZoo)

+ [MOSS: An open-sourced plugin-augmented conversational language model](https://github.com/OpenLMLab/MOSS)

+ [FastChat: An open platform for training, serving, and evaluating large language model based chatbots](https://github.com/lm-sys/FastChat)

+ [LangChain: Building applications with LLMs through composability](https://github.com/hwchase17/langchain)

+ [ChuanhuChatgpt](https://github.com/GaiZhenbiao/ChuanhuChatGPT)
