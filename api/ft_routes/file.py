import os
import secrets

from fastapi import APIRouter, HTTPException, UploadFile, Form, Depends

from api.config import config
from api.routes.utils import check_api_key
from api.utils.protocol import DeleteFileResponse, File, ListFiles

file_router = APIRouter(prefix="/files")


@file_router.post("", response_model=File, dependencies=[Depends(check_api_key)])
async def upload_file(file: UploadFile, purpose: str = Form(...)):
    file_id = f"file-{secrets.token_hex(12)}"
    purpose = purpose.replace("_", "-")
    filename = file.filename
    file_path = os.path.join(config.UPLOAD_FOLDER, f"{file_id}_{purpose}_{filename}")
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return File(
        id=file_id,
        bytes=os.path.getsize(file_path),
        filename=filename,
        purpose=purpose,
    )


@file_router.get("/{id}", response_model=File, dependencies=[Depends(check_api_key)])
async def get_file_info(file_id: str):
    file = _find_file(file_id)
    if file:
        file_id, purpose = file.split("_")[0], file.split("_")[1]
        filename = "_".join(file.split("_")[2:])
        file_path = os.path.join(config.UPLOAD_FOLDER, file)
        return File(
            id=file_id,
            bytes=os.path.getsize(file_path),
            filename=filename,
            purpose=purpose,
            created_at=os.path.getctime(file_path),
        )
    else:
        raise HTTPException(status_code=404, detail=f"File {id} not found!")


@file_router.get("/{id}/content", dependencies=[Depends(check_api_key)])
async def get_file_content(file_id: str):
    file = _find_file(file_id)
    if file:
        file_path = os.path.join(config.UPLOAD_FOLDER, file)
        with open(file_path, "rb") as f:
            content = f.read()
            return content
    else:
        raise HTTPException(status_code=404, detail=f"File {id} not found!")


@file_router.get("", response_model=ListFiles, dependencies=[Depends(check_api_key)])
async def list_files():
    data = []
    for file in os.listdir(config.UPLOAD_FOLDER):
        file_id, purpose = file.split("_")[0], file.split("_")[1]
        filename = "_".join(file.split("_")[2:])
        file_path = os.path.join(config.UPLOAD_FOLDER, file)
        data.append(
            File(
                id=file_id,
                bytes=os.path.getsize(file_path),
                filename=filename,
                purpose=purpose,
                created_at=os.path.getctime(file_path),
            )
        )
    return ListFiles(data=data)


@file_router.delete("/{file_id}", response_model=DeleteFileResponse)
async def delete_file(file_id: str):
    deleted = False
    file = _find_file(file_id)
    if file:
        file_path = os.path.join(config.UPLOAD_FOLDER, file)
        os.remove(file_path)
        deleted = True

    return DeleteFileResponse(id=file_id, deleted=deleted)


def _find_file(file_id: str):
    for file in os.listdir(config.UPLOAD_FOLDER):
        if file.startswith(file_id):
            return file
    return None
