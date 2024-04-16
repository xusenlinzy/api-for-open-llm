from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

text = "你好"
messages = [HumanMessage(content=text)]

llm = ChatOpenAI(openai_api_key="xxx", openai_api_base="http://192.168.20.44:7861/v1")

print(llm.invoke(messages))

embedding = OpenAIEmbeddings(openai_api_key="xxx", openai_api_base="http://192.168.20.44:7861/v1")
print(embedding.embed_documents(["你好"]))
