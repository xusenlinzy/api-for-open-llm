import os
from typing import Optional, Dict

import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sqlalchemy import create_engine
from tqdm import tqdm


def get_table_info(table_name, database, databases):
    import openai

    openai.api_key = "xxx"
    openai.api_base = "http://45.63.96.171:8080/v1"

    db_config = databases[database]
    con = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{database}"
    )
    table_schema = pd.read_sql(f"show create table {table_name};", con=con)["Create Table"][0]
    d = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": table_schema + """\n\n根据上述信息，用一段中文完整准确地描述这张表的含义，并给出一些相关的用户查询问题示例，且保证多样化。"""
            }
        ]
    )
    return d.choices[0].message.content


def get_table_names(database, databases):
    """ 获取数据库表名 """
    if database:
        db_config = databases[database]
        con = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{database}")
        tables = pd.read_sql("show tables;", con=con).values
        return [t[0] for t in tables]


def save_documents(database, databases):
    tables = get_table_names(database, databases)
    documents, metadatas = [], []
    for table in tqdm(tables):
        try:
            documents.append(get_table_info(table, database, databases))
            metadatas.append(table)
            docs = pd.DataFrame({"documents": documents, "metadatas": metadatas})
            docs.to_csv("table_info.csv", index=False)
        except KeyError:
            pass


class SqlChatPipeline:

    def __init__(self, api_base: str, databases: Optional[Dict[str, str]] = None, k: Optional[int] = 3):
        import openai

        self.api_base = api_base
        self.databases = databases or {}
        self.k = k

        openai.api_base = self.api_base
        openai.api_key = "xxx"

        self.chat_client = openai.ChatCompletion

    def add_database(self, user: str, password: str, host: str, port: int, name: str):
        self.databases.update(
            {
                name: {
                    "ueser": user,
                    "password": password,
                    "host": host,
                    "port": port,
                }
            }
        )

    def get_related_tables(self, query: str, database: str, database_info: str, k: Optional[int] = None):
        embeddings = OpenAIEmbeddings(openai_api_key="xxx", openai_api_base=self.api_base)
        vs_path = f"vector_store/sql-{database}"
        if os.path.exists(vs_path):
            vector_store = FAISS.load_local(vs_path, embeddings)
        else:
            df = pd.read_csv(database_info)
            documents = list(df.documents.values)
            metadatas = list({"table_name": t} for t in df.metadatas.values)

            vector_store = FAISS.from_texts(documents, embeddings, metadatas=metadatas)
            vector_store.save_local(f"vector_store/sql-{database}")

        related_docs_with_score = vector_store.similarity_search_with_score(query, k=k or self.k)
        return [i[0].metadata["table_name"] for i in related_docs_with_score]

    def get_table_schema(self, table_name, database):
        db_config = self.databases[database]
        con = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{database}"
        )
        table_schema = pd.read_sql(f"show create table {table_name};", con=con)["Create Table"][0]
        return table_schema

    def generate_messages(self, query: str, database: str, database_info: str, k: Optional[int] = None):
        related_tables = self.get_related_tables(query, database, database_info, k)
        schemas = []
        for table in related_tables:
            try:
                schemas.append(self.get_table_schema(table, database))
            except KeyError:
                pass

        schemas = "\n\n".join(schemas)
        messages = [
            {
                "role": "user",
                "content": f"你现在是一名SQL助手，能够根据用户的问题生成准确的SQL查询。已知相关的建表语句为：\n{schemas}\n根据上述数据库信息，回答相关问题",
            },
            {
                "role": "assistant",
                "content": "好的，我明白了，我会尽可能准确地回答您的问题。",
            },
            {
                "role": "user",
                "content": query,
            }
        ]
        return messages

    def __call__(self, query: str, database: str, database_info: str, k: Optional[int] = None):
        messages = self.generate_messages(query, database, database_info, k)

        ans = self.chat_client.create(
            model="chatglm",
            messages=messages,
            temperature=0.0,
        )
        return ans.choices[0].message.content


if __name__ == "__main__":
    databases = {
        "test2": {
            "user": "root",
            "password": "Dnect_123",
            "host": "192.168.0.121",
            "port": 3306
        }
    }

    sqlchat = SqlChatPipeline("http://192.168.0.53:7890/v1", databases=databases)
    ans = sqlchat("湖北省2022年的项目有哪些", "test2", "table_info.csv")
    print(ans)
