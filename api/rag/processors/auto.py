import importlib
from functools import lru_cache
from typing import (
    Dict,
    Any,
)

import chardet
import langchain_community.document_loaders
from langchain.text_splitter import (
    TextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.base import BaseLoader
from loguru import logger

from api.config import TEXT_SPLITTER_CONFIG

LOADER_MAPPINGS = {
    "UnstructuredHTMLLoader":
        [".html", ".htm"],
    "MHTMLLoader":
        [".mhtml"],
    "TextLoader":
        [".md"],
    "UnstructuredMarkdownLoader":
        [".md"],
    "JSONLoader":
        [".json"],
    "JSONLinesLoader":
        [".jsonl"],
    "CSVLoader":
        [".csv"],
    # "FilteredCSVLoader":
    #     [".csv"], 如果使用自定义分割csv
    "OpenParserPDFLoader":
        [".pdf"],
    "RapidOCRPDFLoader":
        [".pdf"],
    "RapidOCRDocLoader":
        [".docx", ".doc"],
    "RapidOCRPPTLoader":
        [".ppt", ".pptx", ],
    "RapidOCRLoader":
        [".png", ".jpg", ".jpeg", ".bmp"],
    "UnstructuredFileLoader":
        [".eml", ".msg", ".rst", ".rtf", ".txt", ".xml", ".epub", ".odt", ".tsv"],
    "UnstructuredEmailLoader":
        [".eml", ".msg"],
    "UnstructuredEPubLoader":
        [".epub"],
    "UnstructuredExcelLoader":
        [".xlsx", ".xls", ".xlsd"],
    "NotebookLoader":
        [".ipynb"],
    "UnstructuredODTLoader":
        [".odt"],
    "PythonLoader":
        [".py"],
    "UnstructuredRSTLoader":
        [".rst"],
    "UnstructuredRTFLoader":
        [".rtf"],
    "SRTLoader":
        [".srt"],
    "TomlLoader":
        [".toml"],
    "UnstructuredTSVLoader":
        [".tsv"],
    "UnstructuredWordDocumentLoader":
        [".docx", ".doc"],
    "UnstructuredXMLLoader":
        [".xml"],
    "UnstructuredPowerPointLoader":
        [".ppt", ".pptx"],
    "EverNoteLoader":
        [".enex"],
}

SUPPORTED_EXTS = [
    ext for sublist in LOADER_MAPPINGS.values() for ext in sublist
]


class JSONLinesLoader(JSONLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_loader_class(file_extension):
    for cls, exts in LOADER_MAPPINGS.items():
        if file_extension in exts:
            return cls


def get_loader(
    loader_name: str,
    file_path: str,
    loader_kwargs: Dict[str, Any] = None,
) -> BaseLoader:
    """ 根据 loader_name 和文件路径或内容返回文档加载器 """
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in [
            "OpenParserPDFLoader",
            "RapidOCRPDFLoader",
            "RapidOCRLoader",
            "FilteredCSVLoader",
            "RapidOCRDocLoader",
            "RapidOCRPPTLoader",
        ]:
            loaders_module = importlib.import_module(
                "api.rag.processors.loader"
            )
        else:
            loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
        DocumentLoader = getattr(loaders_module, loader_name)

    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)
        loaders_module = importlib.import_module(
            "langchain_community.document_loaders"
        )
        DocumentLoader = getattr(loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)

    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


@lru_cache()
def make_text_splitter(
    splitter_name: str, chunk_size: int, chunk_overlap: int
) -> TextSplitter:
    """ 根据参数获取特定的分词器 """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = TEXT_SPLITTER_CONFIG[splitter_name]["headers_to_split_on"]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
        else:
            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module(
                    "api.rag.processors.splitter"
                )
            except ImportError:  # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )

            TextSplitter = getattr(text_splitter_module, splitter_name)

            if TEXT_SPLITTER_CONFIG[splitter_name]["source"] == "tiktoken":  # 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

            elif TEXT_SPLITTER_CONFIG[splitter_name]["source"] == "huggingface":  # 从huggingface加载
                if TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  # 字符长度加载
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        logger.error(e)
        text_splitter_module = importlib.import_module(
            "langchain.text_splitter"
        )
        TextSplitter = getattr(
            text_splitter_module, "RecursiveCharacterTextSplitter"
        )
        text_splitter = TextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    return text_splitter


