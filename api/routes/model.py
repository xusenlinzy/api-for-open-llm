import time
from typing import List

from fastapi import APIRouter, Depends, status
from openai.types.model import Model
from pydantic import BaseModel

from api.config import SETTINGS
from api.models import LLM_ENGINE
from api.utils.request import check_api_key

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
    res = available_models
    exists = [m.id for m in res.data]
    if SETTINGS.engine == "vllm":
        models = await LLM_ENGINE.show_available_models()
        for m in models.data:
            if m.id not in exists:
                res.data.append(m)
    return res


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
