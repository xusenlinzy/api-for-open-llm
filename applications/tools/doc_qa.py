import os
import shutil
from typing import Optional

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import embed_with_retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
问题是：{question}"""


class CustomEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, chunk_size=0):
        response = embed_with_retry(
            self,
            input=texts,
            engine=self.deployment,
            request_timeout=self.request_timeout,
        )
        return [r["embedding"] for r in response["data"]]


def load_file(filepath, chunk_size=500, chunk_overlap=0):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = loader.load_and_split(text_splitter=text_splitter)
    return docs


def generate_prompt(related_docs, query: str, prompt_template=PROMPT_TEMPLATE) -> str:
    context = "\n".join([doc[0].page_content for doc in related_docs])
    return prompt_template.replace("{question}", query).replace("{context}", context)


class DocQAPromptAdapter:
    def __init__(self, chunk_size: Optional[int] = 500, chunk_overlap: Optional[int] = 0, api_key: Optional[str] = "xxx"):
        self.embeddings = CustomEmbeddings(openai_api_key=api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vector_store = None
        self.vs_path = None

    def create_vector_store(self, file_path, vs_path, embeddings=None):
        docs = load_file(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.vector_store = FAISS.from_documents(docs, self.embeddings if not embeddings else embeddings)
        self.vector_store.save_local(vs_path)

    def reset_vector_store(self, vs_path, embeddings=None):
        self.vector_store = FAISS.load_local(vs_path, self.embeddings if not embeddings else embeddings)

    @staticmethod
    def delete_files(files):
        for file in files:
            if os.path.exists(file):
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)

    def __call__(self, query, vs_path=None, topk=6):
        if vs_path is not None and os.path.exists(vs_path):
            self.reset_vector_store(vs_path)
        self.vector_store.embedding_function = self.embeddings.embed_query
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=topk)
        return generate_prompt(related_docs_with_score, query)
