import time
from typing import List

from fastapi import APIRouter, Depends, status
from openai.types.model import Model
from pydantic import BaseModel

from api.config import SETTINGS
from api.utils import check_api_key

model_router = APIRouter()


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model] = []


available_models = ModelList(
    data=[
        Model(
            id=name,
            object="model",
            created=int(time.time()),
            owned_by="open"
        )
        for name in SETTINGS.model_names if name
    ]
)


@model_router.get(
    "/models",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def show_available_models():
    return available_models


@model_router.get(
    "/models/{model}",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def retrieve_model():
    return Model(
        id=model,
        object="model",
        created=int(time.time()),
        owned_by="open"
    )
