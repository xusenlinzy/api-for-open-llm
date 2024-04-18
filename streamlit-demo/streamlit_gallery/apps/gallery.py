import streamlit as st


def main():
    text = """# Welcome to Api for Open LLM!

## 环境依赖

```shell
openai
langchain
langchain_openai
python-dotenv==1.0.0
loguru
sqlalchemy~=1.4.46
spacy
lancedb
pymysql
streamlit
google-search-results
jupyter_client

```

## 环境变量解释

+ `CHAT_API_BASE`: 聊天接口地址，例如：`http://192.168.0.53:7891/v1`


+ `SQL_CHAT_API_BASE`: `sql` 生成模型接口地址（可选）


+ `TOOL_CHAT_API_BASE`: 调用工具模型接口地址（可选）


+ `EMBEDDING_API_BASE`: 嵌入模型接口地址（可选）


+ `API_KEY`: 默认不需要配置


+ `SERPAPI_API_KEY`: 搜索功能需要


+ `IPYKERNEL`: `python` 解释器名称


+ `INTERPRETER_CHAT_API_BASE`: 代码解释器模型接口地址（可选）


## 支持对话模式

|        模式        |     含义     |        状态        |
|:----------------:|:----------:|:----------------:|
|       Chat       |   普通聊天模式   | `👷 Development` |
|     Doc Chat     |   文档问答模式   | `👷 Development` |
|     SQL Chat     | `SQL` 生成模式 | `👷 Development` |
|    Tool Chat     |   工具调用模式   |   `🧪 Testing`   |
| Code Interpreter |  代码解释器模式   |   `🧪 Testing`   |
|      Agent       |   代理调用模式   |  `👟 Planning`   |


"""
    st.markdown(text)


if __name__ == "__main__":
    main()
