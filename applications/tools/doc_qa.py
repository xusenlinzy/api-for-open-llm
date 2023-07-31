import os
import shutil
from typing import Optional

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger
from tqdm import tqdm

from .parser import parse_pdf

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
问题是：{question}"""


def _get_documents(filepath, chunk_size=500, chunk_overlap=0, two_column=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    file_type = os.path.splitext(filepath)[1]

    logger.info(f"Loading file: {filepath}")
    texts = Document(page_content="", metadata={"source": filepath})
    try:
        if file_type == ".pdf":
            logger.debug("Loading PDF...")
            try:
                pdftext = parse_pdf(filepath, two_column).text
            except:
                from PyPDF2 import PdfReader

                pdftext = ""
                with open(filepath, "rb") as pdfFileObj:
                    pdfReader = PdfReader(pdfFileObj)
                    for page in tqdm(pdfReader.pages):
                        pdftext += page.extract_text()

            texts = [Document(page_content=pdftext, metadata={"source": filepath})]

        elif file_type == ".docx":
            from langchain.document_loaders import UnstructuredWordDocumentLoader

            logger.debug("Loading Word...")
            loader = UnstructuredWordDocumentLoader(filepath)
            texts = loader.load()
        elif file_type == ".pptx":
            from langchain.document_loaders import UnstructuredPowerPointLoader

            logger.debug("Loading PowerPoint...")
            loader = UnstructuredPowerPointLoader(filepath)
            texts = loader.load()
        elif file_type == ".epub":
            from langchain.document_loaders import UnstructuredEPubLoader

            logger.debug("Loading EPUB...")
            loader = UnstructuredEPubLoader(filepath)
            texts = loader.load()
        elif file_type == ".md":
            loader = UnstructuredFileLoader(filepath, mode="elements")
            return loader.load()
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            return loader.load_and_split(text_splitter=text_splitter)
    except Exception as e:
        import traceback
        logger.error(f"Error loading file: {filepath}")
        traceback.print_exc()

    return text_splitter.split_documents(texts)


def get_documents(filepath, chunk_size=500, chunk_overlap=0, two_column=False):
    documents = []
    logger.debug("Loading documents...")
    if os.path.isfile(filepath):
        documents.extend(
            _get_documents(
                filepath,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                two_column=two_column
            )
        )
    else:
        for file in filepath:
            documents.extend(
                _get_documents(
                    file,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    two_column=two_column
                )
            )
    logger.debug("Documents loaded.")
    return documents


def generate_prompt(related_docs, query: str, prompt_template=PROMPT_TEMPLATE) -> str:
    context = "\n".join([doc[0].page_content for doc in related_docs])
    return prompt_template.replace("{question}", query).replace("{context}", context)


class DocQAPromptAdapter:
    def __init__(self, chunk_size: Optional[int] = 500, chunk_overlap: Optional[int] = 0, api_key: Optional[str] = "xxx"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vector_store = None

    def create_vector_store(self, file_path, vs_path, embeddings=None):
        documents = get_documents(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.vector_store = FAISS.from_documents(documents, self.embeddings if not embeddings else embeddings)
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
