import os

os.environ["OPENAI_API_BASE"] = "http://192.168.0.53:7891/v1"
os.environ["OPENAI_API_KEY"] = "xxx"

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.embeddings import OpenAIEmbeddings

chat = ChatOpenAI()
print(chat([HumanMessage(content="你好")]))


embeddings = OpenAIEmbeddings()
query_result = embeddings.embed_query("什么是chatgpt？")
print(query_result)
