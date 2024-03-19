import importlib
import os
from typing import List

from langchain_community.vectorstores import FAISS
from loguru import logger
from sentence_transformers import SentenceTransformer

LOADER_DICT = {
    "UnstructuredFileLoader": [
        '.eml', '.html', '.json', '.md', '.msg', '.rst',
        '.rtf', '.txt', '.xml', '.doc', '.docx', '.epub',
        '.odt', '.pdf', '.ppt', '.pptx', '.tsv'
    ],
    "CSVLoader": [".csv"],
    "PyPDFLoader": [".pdf"],
}
SUPPORTED_EXTS = set([ext for sublist in LOADER_DICT.values() for ext in sublist])


def get_loader_class(file_extension):
    for c, exts in LOADER_DICT.items():
        if file_extension in exts:
            return c


def load_document(filepath, chunk_size: int = 300, chunk_overlap: int = 10):
    """ 加载文档 """
    ext = os.path.splitext(filepath)[-1].lower()
    document_loader_name = get_loader_class(ext)
    try:
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, document_loader_name)
    except Exception as e:
        logger.warning(e)
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if document_loader_name == "UnstructuredFileLoader":
        loader = DocumentLoader(filepath, autodetect_encoding=True)
    else:
        loader = DocumentLoader(filepath)

    try:
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "SpacyTextSplitter")
        text_splitter = TextSplitter(
            pipeline="zh_core_web_sm",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        logger.warning(e)
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    docs = loader.load_and_split(text_splitter)
    return docs


class Embeddings:
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        encod_list = embeddings.tolist()
        return encod_list

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class FaissDocServer:
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = None
        self.vs_path = None

    def doc_upload(self, file_path: str, chunk_size: int, chunk_overlap: int, vs_path: str):
        if not os.path.exists(vs_path):
            documents = load_document(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.vector_store = FAISS.from_documents(documents, self.embedding)
            self.vs_path = vs_path
            self.vector_store.save_local(vs_path)
        logger.info("Successfully inserted documents!")

    def doc_search(self, query: str, top_k: int, vs_path: str) -> List:
        if vs_path != self.vs_path and os.path.exists(vs_path):
            self.vector_store = FAISS.load_local(vs_path, self.embedding)
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=top_k)
        return related_docs_with_score


DOCQA_PROMPT = """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>

<已知信息>{context}</已知信息>

<问题>{query}</问题>"""
