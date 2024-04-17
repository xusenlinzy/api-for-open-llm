import os
import secrets
import uuid
from pathlib import Path
from typing import List, Optional

import lancedb
import pyarrow as pa
import requests
from lancedb.rerankers import CohereReranker
from loguru import logger
from openai import OpenAI

EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
CO_API_URL = os.getenv("CO_API_URL")

if not CO_API_URL:
    os.environ["CO_API_URL"] = EMBEDDING_API_BASE


class RefinedCohereReranker(CohereReranker):
    def _rerank(self, result_set: pa.Table, query: str):
        docs = result_set[self.column].to_pylist()
        results = self._client.rerank(
            query=query,
            documents=docs,
            top_n=self.top_n,
            model=self.model_name,
        )  # returns list (text, idx, relevance) attributes sorted descending by score
        indices, scores = list(
            zip(*[(result.index, result.relevance_score) for result in results.results])
        )  # tuples
        result_set = result_set.take(list(indices))
        # add the scores
        result_set = result_set.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )

        return result_set


class DocServer:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.vs_path = None
        self.client = OpenAI(
            base_url=os.getenv("EMBEDDING_API_BASE"),
            api_key=os.getenv("API_KEY", ""),
        )
        self.db = lancedb.connect(
            os.path.join(Path(__file__).parents[3], "lancedb.db"),
        )

    def upload(
        self,
        filepath: Optional[str] = None,
        url: Optional[str] = None,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        table_name: str = None,
    ) -> str:
        if url is not None:
            data = {
                "url": url,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            file_id = str(secrets.token_hex(12))
        else:
            upf = self.client.files.create(file=open(filepath, "rb"), purpose="assistants")
            file_id, filename = upf.id, upf.filename
            data = {
                "file_id": file_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

        res = requests.post(
            url=os.getenv("EMBEDDING_API_BASE") + "/files/split", json=data,
        ).json()

        table_name = table_name or file_id
        embeddings = self.embeddings.embed_documents(
            [doc["page_content"] for doc in res["docs"]]
        )
        data = []
        for i, doc in enumerate(res["docs"]):
            append_data = {
                "id": str(uuid.uuid4()),
                "vector": embeddings[i],
                "text": doc["page_content"],
                "metadata": doc["metadata"]["source"],
            }
            data.append(append_data)

        if table_name in self.db.table_names():
            tbl = self.db.open_table(table_name)
            tbl.add(data)
        else:
            self.db.create_table(table_name, data)

        logger.info("Successfully inserted documents!")

        return table_name

    def search(
        self,
        query: str,
        top_k: int,
        table_name: str,
        rerank: bool = False,
    ) -> List:
        table = self.db.open_table(table_name)
        embedding = self.embeddings.embed_query(query)
        top_n = 2 * top_k if rerank else top_k

        docs = table.search(
            embedding,
            vector_column_name="vector",
        ).metric("cosine").limit(top_n)

        if rerank:
            docs = docs.rerank(
                reranker=RefinedCohereReranker(api_key="xxx"),
                query_string=query,
            ).limit(top_k)

        docs = docs.to_pandas()
        del docs["vector"]
        del docs["id"]
        return docs

    def delete(self, table_name):
        if table_name:
            self.db.drop_table(table_name)
            try:
                self.client.files.delete(file_id=table_name)
            except:
                pass
        return table_name


DOCQA_PROMPT = """参考信息：
{context}
---
我的问题或指令：
{query}
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""
