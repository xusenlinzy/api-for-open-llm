import os
import secrets
from typing import (
    List,
    Optional,
    Any,
)

import requests
from fastapi import APIRouter, HTTPException, UploadFile
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from openai.pagination import SyncPage
from openai.types.file_deleted import FileDeleted
from openai.types.file_object import FileObject
from pydantic import BaseModel

from api.config import STORAGE_LOCAL_PATH
from api.rag.processors import (
    get_loader,
    make_text_splitter,
    get_loader_class,
)
from api.rag.processors.splitter import zh_title_enhance as func_zh_title_enhance

file_router = APIRouter(prefix="/files")


class File2DocsRequest(BaseModel):
    file_id: Optional[str] = None
    url: Optional[str] = None
    zh_title_enhance: Optional[bool] = False
    chunk_size: Optional[int] = 250
    chunk_overlap: Optional[int] = 50
    text_splitter_name: Optional[str] = "ChineseRecursiveTextSplitter"
    url_parser_prefix: Optional[str] = "https://r.jina.ai/"


class File2DocsResponse(BaseModel):
    id: str
    object: str = "docs"
    docs: List[Any]


@file_router.post("", response_model=FileObject)
async def upload_file(file: UploadFile):
    file_id = "file-" + str(secrets.token_hex(12)).replace("-", "_")
    filename = file.filename
    filepath = os.path.join(STORAGE_LOCAL_PATH, f"{file_id}_{filename}")
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return FileObject(
        id=file_id,
        bytes=os.path.getsize(filepath),
        created_at=int(os.path.getctime(filepath)),
        filename=filename,
        object="file",
        purpose="assistants",
        status="uploaded",
    )


@file_router.get("/{file_id}", response_model=FileObject)
async def get_details(file_id: str):
    file = _find_file(file_id)
    if file:
        file_id = file.split("_")[0]
        filename = "_".join(file.split("_")[1:])
        filepath = os.path.join(STORAGE_LOCAL_PATH, file)
        return FileObject(
            id=file_id,
            bytes=os.path.getsize(filepath),
            created_at=int(os.path.getctime(filepath)),
            filename=filename,
            object="file",
            purpose="assistants",
            status="uploaded",
        )
    else:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found!")


@file_router.get("")
async def list_files():
    data = []
    for file in os.listdir(STORAGE_LOCAL_PATH):
        file_id = file.split("_")[0]
        filename = "_".join(file.split("_")[1:])
        filepath = os.path.join(STORAGE_LOCAL_PATH, file)
        data.append(
            FileObject(
                id=file_id,
                bytes=os.path.getsize(filepath),
                created_at=int(os.path.getctime(filepath)),
                filename=filename,
                object="file",
                purpose="assistants",
                status="uploaded",
            )
        )
    return SyncPage(data=data, object="list")


@file_router.delete("/{file_id}", response_model=FileDeleted)
async def delete_file(file_id: str):
    deleted = False
    filename = _find_file(file_id)
    if filename:
        filepath = os.path.join(STORAGE_LOCAL_PATH, filename)
        if filepath:
            os.remove(filepath)
            deleted = True
        return FileDeleted(id=file_id, object="file", deleted=deleted)
    else:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found!")


@file_router.post("/split", response_model=File2DocsResponse)
async def split_into_docs(request: File2DocsRequest):
    if request.url is not None:
        # https://github.com/jina-ai/reader
        try:
            headers = {"Accept": "application/json"}
            res = requests.get(f"{request.url_parser_prefix}{request.url}", headers=headers).json()
            docs = [
                Document(page_content=res["data"]["content"])
            ]
            ext = ""
            source = {"url": request.url, "title": res["data"].get("title")}
        except:
            raise HTTPException(status_code=404, detail=f"Parsing {request.url} failed!")

    else:
        filename = _find_file(request.file_id)
        if not filename:
            raise HTTPException(status_code=404, detail=f"File {request.file_id} not found!")

        filepath = os.path.join(STORAGE_LOCAL_PATH, filename)
        ext = os.path.splitext(filepath)[-1].lower()
        loader = get_loader(loader_name=get_loader_class(ext), file_path=filepath)

        if isinstance(loader, TextLoader):
            loader.encoding = "utf8"
            docs = loader.load()
        else:
            docs = loader.load()

        source = {
            "file_id": request.file_id,
            "filename": "_".join(filename.split("_")[1:])
        }

    if not docs:
        return []

    if ext not in [".csv"]:
        text_splitter = make_text_splitter(
            splitter_name=request.text_splitter_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        if request.text_splitter_name == "MarkdownHeaderTextSplitter":
            docs = text_splitter.split_text(docs[0].page_content)
        else:
            docs = text_splitter.split_documents(docs)

    if not docs:
        return []

    if request.zh_title_enhance:
        docs = func_zh_title_enhance(docs)

    return File2DocsResponse(
        id="docs-" + str(secrets.token_hex(12)),
        docs=[
            dict(
                page_content=d.page_content,
                metadata={"source": source},
                type="Document",
            )
            for d in docs
        ]
    )


def _find_file(file_id: str):
    files = os.listdir(STORAGE_LOCAL_PATH)
    for file in files:
        if file.startswith(file_id):
            return file
    return None
